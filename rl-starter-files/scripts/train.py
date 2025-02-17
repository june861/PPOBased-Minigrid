import argparse
import time
import wandb
import datetime
import torch_ac
import tensorboardX
import sys
import numpy as np
import utils
from utils import device
from model import ACModel

# Parse arguments
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=500,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
 
# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=128,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

# DONE(junweiluo): 增加需要的参数
parser.add_argument("--use_action_dist", action="store_true", default=False, help="use extra action to adjust ratio")
parser.add_argument("--eval_freq", type=int, default=5, help="the evaluation freqency for every steps")
parser.add_argument("--eval_times", type=int, default=32, help="the evaluation times, recommend it as 16'times ")
parser.add_argument("--use_surr", action="store_true", default=False, help="use 4surr")
parser.add_argument("--sample_all_act", action="store_true", default=False, help="sample all other action to adjust ratio")
parser.add_argument("--use_noise", action="store_true", default=False, help="noise contrasitive experiment")
parser.add_argument("--use_wandb", action="store_false", default=True, help="use wandb tool to log")
parser.add_argument("--debugger_mode", action="store_true", default=False, help="use launch.json to launch debgguer mode")

if __name__ == "__main__":
    args = parser.parse_args()
    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    if args.use_wandb:
        if args.use_action_dist == False:
            if args.use_noise == False:
                name_ = 'baseline'
            else:
                name_ = 'noise'
        else:
            if args.sample_all_act:
                if args.use_surr:
                    name_ = "all_surr"
                else:
                    name_ = "all"
            else:
                if args.use_surr:
                    name_ = "two_surr"
                else:
                    name_ = "two"

        times_ = int(time.time())
        if args.debugger_mode:
            group = f'{args.env}_{name_}_debugger'
        else:
            group = f'{args.env}_{name_}'
        wandb_name = f'{args.env}_{name_}_{args.seed}_{times_}'
        wandb.init(
            project = "ppo-minigrid-v1",
            group = group,
            name = wandb_name,
            sync_tensorboard = True,
            monitor_gym=True,
            config=vars(args),
            save_code=False,
        )
    
    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        # DONE(junweiluo): 增加调用参数
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, 
                                action_dim = int(envs[0].action_space.n), use_action_dist = args.use_action_dist,
                                use_surr4 = args.use_surr, sample_all_act = args.sample_all_act, use_noise = args.use_noise)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    eval_steps = 0
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm","ratio","ratio1", "ratio2"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["ratio"],logs["ratio1"], logs["ratio2"]]

            # DONE(junweiluo)： 修改写入Tensorboard的指标
            add_to_scalars = {
                "train_reward": rreturn_per_episode['mean'],
                # 'dist_entropy' : logs['entropy'],
                # 'value': logs['value'],
                # 'policy_loss' : logs['policy_loss'],
                # 'value_loss' : logs['value_loss'],
                # 'grad_norm' : logs['grad_norm'],
                # 'ratio1' : logs['ratio1'],
                # 'ratio2' : logs['ratio2'],
            }

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            if args.use_wandb:
                # build dict
                logs_ = dict()
                for field, value in zip(header, data):
                    logs_[field] = value
                wandb.log(logs_)

            for k, v in add_to_scalars.items():
                tb_writer.add_scalar(k, v, num_frames)

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # DONE(junweiluo)：增加训练时评估
        # if update % args.eval_freq == 0:
        #     eval_epoch = args.eval_times // args.procs
        #     eval_total_reward = 0.0
        #     for _ in range(eval_epoch):
        #         eval_exps, eval_logs = algo.collect_experiences()
        #         eval_total_reward += np.mean(logs1['return_per_episode']).item()
        #     eval_total_reward /= eval_epoch
        #     tb_writer.add_scalar('eval_avg_reward', eval_total_reward, global_step = eval_steps)
        #     eval_steps += 1


        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
