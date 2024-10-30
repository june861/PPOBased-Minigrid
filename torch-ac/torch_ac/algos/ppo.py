import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    # DONE(junweiluo)： 增加action_dim作为参数用于初始化BaseAlgo
    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=5, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, action_dim = None, use_action_dist = True):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, action_dim)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        # DONE(junweiluo) : 增加参数，用于区分baseline
        self._use_action_dist = use_action_dist

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            # DONE(junweiluo) : 增加两个ratio的指标
            log_ratio1 = []
            log_ratio2 = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_ratio1 = 0.0
                batch_ratio2 = 0.0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    # DONE(junweiluo) : PPO改进算法
                    if self._use_action_dist:
                        ratio1 = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                        # 随机采样一个额外的动作
                        a_logprobs = dist.logits
                        mask_ = torch.ones_like(sb.a_logprobs)
                        mask_[torch.arange(sb.a_logprobs.shape[0]), sb.action] = 0.0
                        selected_indice = torch.multinomial(mask_, num_samples=1).squeeze()
                        old_a_logprobs_2 = sb.a_logprobs[torch.arange(sb.a_logprobs.shape[0]), selected_indice]
                        a_logprobs_2 = a_logprobs[torch.arange(sb.a_logprobs.shape[0]), selected_indice]
                        # 去梯度
                        ratio2 = torch.exp(a_logprobs_2 - old_a_logprobs_2).detach()
                        ratio2 = torch.clamp(ratio2, 1.0 - self.clip_eps / 2, 1.0 + self.clip_eps / 2)
                        ratio = ratio1 * ratio2
                    
                    # baseline
                    else:
                        ratio1 = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                        ratio2 = torch.ones_like(ratio1).detach()
                        ratio = ratio1
                    
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss
                    batch_ratio1 += ratio1.mean().item()
                    batch_ratio2 += ratio2.mean().item()

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                # DONE(junweiluo): add new metric
                batch_ratio1 /= self.recurrence
                batch_ratio2 /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                # DONE(junweiluo): add new metric
                log_ratio1.append(batch_ratio1)
                log_ratio2.append(batch_ratio2)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "ratio1" : numpy.mean(log_ratio1),
            "ratio2" : numpy.mean(log_ratio2),
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes