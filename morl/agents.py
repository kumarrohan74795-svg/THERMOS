import numpy as np
import torch
import copy
import torch.nn.functional as F
from torch.distributions import Categorical
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.utils import probs_to_logits


class MOPPOAgent(nn.Module):
    def __init__(self, actor, critic, args):
        super(MOPPOAgent, self).__init__()

        self.preference = None
        self.w_ep = None
        self.w_batch = []
        self.actor = actor
        self.critic = critic
        self.args = args
        
        # self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate_critic, eps=1e-5)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate_actor, eps=1e-5)

    def get_value(self, x, preference):
        x = torch.cat((x, preference), dim=1)
        return self.critic(x)

    def get_action_and_value(self, x, preference = None, action=None, deterministic= False, mask = []):

        if preference is None:
            if self.w_ep is None:
                self.w_ep = (torch.tensor(self.w_batch[np.random.choice(np.arange(0,len(self.w_batch))),:])).type(torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor)
                self.w_ep = torch.round(self.w_ep*(10**1))/(10**1)

            preference = self.w_ep   

            x = torch.cat((x.unsqueeze(0), preference.unsqueeze(0)), dim=1)
        else:
            x = torch.cat((x, preference), dim=1).to(self.args.device)
        if len(mask) == 0:
            probs = Categorical(logits=probs_to_logits(self.actor(x,deterministic)))

            if not deterministic:
                if action is None:
                    action = probs.sample()

                # valid_actions - logits
                valid_actions = torch.from_numpy(probs.probs.squeeze(0).cpu().numpy())

                return action, probs.log_prob(action), probs.entropy(), self.critic(x), valid_actions
            else:
                action_probabiltiies = probs.probs.cpu().numpy()
                action = np.argmax(action_probabiltiies)

                valid_actions = torch.from_numpy(probs.probs.squeeze(0).cpu().numpy())

                return action, valid_actions

        else: 
            logits = probs_to_logits(self.actor(x,deterministic))
            if not torch.is_tensor(mask):
                valid_actions = torch.BoolTensor(logits.squeeze(0).shape)*False
                valid_actions[mask] = True
                logits = torch.where(valid_actions.to(self.args.device), logits, torch.tensor(-1e8).to(self.args.device))
            else:
                logits = torch.where(mask, logits, torch.tensor(-1e8).to(self.args.device))
                valid_actions = mask
            probs = Categorical(logits=logits)
            if not deterministic:
                if action is None:
                    action = probs.sample()
               
                return action, probs.log_prob(action), probs.entropy(), self.critic(x), valid_actions
            else:
                action_probabiltiies = probs.probs.cpu().numpy()
                action = np.argmax(action_probabiltiies)
                return action, valid_actions

    def learn(self, writer, global_step, batch_values, b_masks = []):

        if len(b_masks) == 0:
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_preferences, b_masks = batch_values
        else:
            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_preferences = batch_values
        
        preferences_batch_critic = torch.FloatTensor(torch.from_numpy(copy.deepcopy(b_preferences.cpu().numpy()))).to(self.args.device)

        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        entropy_losses = []

        for epoch in range(self.args.update_epochs):
            
            approx_kl_divs = []
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                if len(b_masks) == 0:
                    _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_preferences[mb_inds], b_actions[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(b_obs[mb_inds], b_preferences[mb_inds], b_actions[mb_inds], mask = b_masks[mb_inds].bool())
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kl_divs.append(approx_kl.detach().numpy())
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = torch.bmm(b_preferences[mb_inds].unsqueeze(1),mb_advantages.unsqueeze(2)).squeeze()
                if self.args.norm_adv:
                    mb_advantages =(mb_advantages - torch.mean(mb_advantages)) / (torch.std(mb_advantages) + 1e-8) #(axis=0)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() 

                # Value loss
                # newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # v_loss_unclipped = F.smooth_l1_loss(newvalue,b_returns[mb_inds])
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    # v_loss_clipped =  F.smooth_l1_loss(v_clipped,b_returns[mb_inds])
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = v_loss_max.mean()
                else:
                    angle_term_critic = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(preferences_batch_critic[mb_inds]-1,newvalue),0, 0.9999)))
                    v_loss = F.smooth_l1_loss(newvalue,b_returns[mb_inds])

                entropy_loss = entropy.mean()
                entropy_losses.append(entropy_loss.detach().numpy())

                critic_loss = v_loss * self.args.vf_coef
                actor_loss = pg_loss - self.args.ent_coef * entropy_loss

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                self.actor_optimizer.step()
            
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true,axis=0)
        error_var = np.var(y_true - y_pred,axis=0)
        explained_var= np.zeros((self.args.reward_size))
        for i in range(self.args.reward_size):
            explained_var[i] = np.nan if var_y[i] == 0 else 1 - error_var[i] / var_y[i]


        grad_norm_actor = nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        grad_norm_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("Parameters/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("Parameters/learning_rate", self.critic_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("Loss/value_loss", v_loss.item(), global_step)
        writer.add_scalar("Loss/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("Loss/CosineAngle_Critic", angle_term_critic.mean().item(), global_step)
        writer.add_scalar("Loss/entropy", np.mean(entropy_losses), global_step)
        writer.add_scalar("Loss/approx_kl", np.mean(approx_kl_divs), global_step)
        writer.add_scalar("Loss/clipfrac", np.mean(clipfracs), global_step)

       
        for i in range(self.args.reward_size):
            writer.add_scalar("Loss/explained_variance{}".format(i), explained_var[i], global_step)
        writer.add_scalar('Gradient Norm/Actor', grad_norm_actor, global_step)
        writer.add_scalar('Gradient Norm/Critic', grad_norm_critic, global_step)
