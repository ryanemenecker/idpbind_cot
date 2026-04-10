import torch

class FIRE(torch.optim.Optimizer):
    def __init__(self, params, dt_init=0.01, dt_max=0.1, max_step=1.0,
                 N_min=5, f_inc=1.1, f_dec=0.5, alpha_start=0.1, f_alpha=0.99):
        defaults = dict(dt=dt_init, dt_max=dt_max, max_step=max_step,
                        N_min=N_min, f_inc=f_inc, f_dec=f_dec, 
                        alpha_start=alpha_start, f_alpha=f_alpha)
        super(FIRE, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Force is negative gradient
                F = -p.grad
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p)
                    state['dt'] = group['dt']
                    state['alpha'] = group['alpha_start']
                    state['N_since_negative'] = 0

                v = state['v']
                dt = state['dt']
                alpha = state['alpha']

                # Calculate Power: P = F * v
                # We flatten the tensors to take the global dot product
                P = torch.dot(F.flatten(), v.flatten())

                # FIRE rule: check P, quench/update dt BEFORE the MD step so that
                # after a quench the next step starts from v=0 and immediately
                # accumulates v=F*dt, giving P>0 on the following step.
                if P > 0:
                    state['N_since_negative'] += 1
                    if state['N_since_negative'] > group['N_min']:
                        dt = min(dt * group['f_inc'], group['dt_max'])
                        alpha = alpha * group['f_alpha']
                elif P < 0:
                    state['N_since_negative'] = 0
                    dt = dt * group['f_dec']
                    alpha = group['alpha_start']
                    v.zero_()  # The Quench! (in-place so state tensor is cleared)

                # Steer velocity towards the force vector
                v_norm = torch.norm(v)
                F_norm = torch.norm(F)
                if F_norm > 0:
                    v = (1.0 - alpha) * v + alpha * v_norm * (F / F_norm)

                v += F * dt

                # Cap the maximum coordinate change per step to prevent explosions.
                # Critically: also rescale the stored velocity to match the actual
                # displacement taken.  Without this, velocity accumulates unchecked
                # while positions are clamped, causing the optimizer to diverge.
                step_displacement = v * dt
                disp_norm = torch.norm(step_displacement, dim=-1, keepdim=True)
                scale = torch.clamp(group['max_step'] / (disp_norm + 1e-8), max=1.0)
                step_displacement = step_displacement * scale
                v = v * scale  # keep velocity consistent with actual movement

                # Position integration
                p.add_(step_displacement)

                # Update state
                state['v'] = v
                state['dt'] = dt
                state['alpha'] = alpha
                state['step'] += 1

        return loss