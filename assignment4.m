% 1D Heat Equation Solver Comparison
% Parameters
L = 1.0;            % Length (m)
T_init = 350.0;     % Initial Temp (K)
T_left = 300.0;     % BC at x=0
T_right = 400.0;    % BC at x=L
dx = 0.01;          % Spatial step
x = 0:dx:L;
Nx = length(x);
alpha_values = [0.0001, 0.001, 0.01];
check_times = [1.0, 5.0, 10.0, 50.0, 100.0];

for alpha = alpha_values
    fprintf('\nProcessing Alpha = %.4f\n', alpha);
    
    % --- Timestep Logic ---
    dt_exp = 0.5 * dx^2 / alpha * 0.95;
    dt_imp = 0.05; % Implicit is stable, can use larger steps
    
    % --- 1. Explicit Solver ---
    res_exp = solve_explicit(alpha, dt_exp, check_times, x, T_init, T_left, T_right);
    
    % --- 2. Implicit Solver ---
    res_imp = solve_implicit(alpha, dt_imp, check_times, x, T_init, T_left, T_right);
    
    % --- 3. Crank-Nicolson Solver ---
    res_cn = solve_cn(alpha, dt_imp, check_times, x, T_init, T_left, T_right);
    
    % --- 4. MATLAB pdepe Solver ---
    m = 0; % Slab geometry
    sol_pdepe = pdepe(m, @(x,t,u,dudx) heatpde(x,t,u,dudx,alpha), ...
                      @(x) T_init, ...
                      @(xl,ul,xr,ur,t) heatbc(xl,ul,xr,ur,t,T_left,T_right), ...
                      x, check_times);
    
    % --- Plotting ---
    figure('Position', [100, 100, 1200, 400]);
    sgtitle(['1D Heat Equation - \alpha = ', num2str(alpha), ' m^2/s']);
    
    for i = 1:length(check_times)
        subplot(1, 5, i);
        t_curr = check_times(i);
        
        % Analytical
        Y_ana = analytical_solution(x, t_curr, alpha, L, T_left, T_right, T_init);
        
        plot(x, res_exp(i,:), 'r-', 'LineWidth', 1.5); hold on;
        plot(x, res_imp(i,:), 'b--', 'LineWidth', 1.5);
        plot(x, res_cn(i,:), 'g:', 'LineWidth', 2);
        plot(x, sol_pdepe(i,:), 'm-.', 'LineWidth', 1); % pdepe result
        plot(x, Y_ana, 'ko', 'MarkerSize', 3);
        
        title(['Time = ', num2str(t_curr), 's']);
        if i == 1
            ylabel('Temperature (K)');
            legend('Exp', 'Imp', 'CN', 'pdepe', 'Analytic');
        end
        grid on;
    end
end

% --- Helper Functions ---

function T_res = solve_explicit(alpha, dt, check_times, x, T_init, T_L, T_R)
    Nx = length(x); dx = x(2)-x(1);
    r = alpha * dt / dx^2;
    T = ones(1, Nx) * T_init;
    T(1) = T_L; T(end) = T_R;
    T_res = zeros(length(check_times), Nx);
    curr_t = 0; idx = 1;
    
    while idx <= length(check_times)
        if curr_t >= check_times(idx)
            T_res(idx,:) = T;
            idx = idx + 1;
        end
        T_new = T;
        T_new(2:end-1) = T(2:end-1) + r*(T(1:end-2) - 2*T(2:end-1) + T(3:end));
        T = T_new;
        curr_t = curr_t + dt;
    end
end

function T_res = solve_implicit(alpha, dt, check_times, x, T_init, T_L, T_R)
    Nx = length(x); dx = x(2)-x(1);
    r = alpha * dt / dx^2;
    M = Nx - 2;
    T = ones(M, 1) * T_init;
    
    % Build Tridiagonal Matrix
    main = (1 + 2*r) * ones(M, 1);
    off = -r * ones(M-1, 1);
    A = diag(main) + diag(off, 1) + diag(off, -1);
    
    T_res = zeros(length(check_times), Nx);
    curr_t = 0; idx = 1;
    while idx <= length(check_times)
        if curr_t >= check_times(idx)
            T_res(idx,:) = [T_L; T; T_R];
            idx = idx + 1;
        end
        b = T;
        b(1) = b(1) + r*T_L;
        b(end) = b(end) + r*T_R;
        T = A\b;
        curr_t = curr_t + dt;
    end
end

function T_res = solve_cn(alpha, dt, check_times, x, T_init, T_L, T_R)
    Nx = length(x); dx = x(2)-x(1);
    r = alpha * dt / dx^2;
    M = Nx - 2;
    T = ones(M, 1) * T_init;
    
    % Crank-Nicolson Matrices
    AL = diag((1+r)*ones(M,1)) + diag(-r/2*ones(M-1,1),1) + diag(-r/2*ones(M-1,1),-1);
    AR = diag((1-r)*ones(M,1)) + diag(r/2*ones(M-1,1),1) + diag(r/2*ones(M-1,1),-1);
    
    T_res = zeros(length(check_times), Nx);
    curr_t = 0; idx = 1;
    while idx <= length(check_times)
        if curr_t >= check_times(idx)
            T_res(idx,:) = [T_L; T; T_R];
            idx = idx + 1;
        end
        b = AR * T;
        b(1) = b(1) + r/2*(T_L + T_L); % current + next BC
        b(end) = b(end) + r/2*(T_R + T_R);
        T = AL\b;
        curr_t = curr_t + dt;
    end
end

function T = analytical_solution(x, t, alpha, L, T_L, T_R, T_0)
    T_ss = T_L + (T_R - T_L) * x / L;
    v = zeros(size(x));
    for n = 1:100
        kn = n * pi / L;
        % Bn integration for T(x,0) - T_ss
        Bn = (2/L) * ( (T_0-T_L)*(1-cos(n*pi))/kn - (T_R-T_L)*(-L*cos(n*pi)/kn) );
        v = v + Bn * sin(kn * x) * exp(-alpha * kn^2 * t);
    end
    T = T_ss + v;
end

% --- pdepe specific functions ---
function [c,f,s] = heatpde(x,t,u,dudx,alpha)
    c = 1/alpha; f = dudx; s = 0;
end

function [pl,ql,pr,qr] = heatbc(xl,ul,xr,ur,t,TL,TR)
    pl = ul - TL; ql = 0;
    pr = ur - TR; qr = 0;
end