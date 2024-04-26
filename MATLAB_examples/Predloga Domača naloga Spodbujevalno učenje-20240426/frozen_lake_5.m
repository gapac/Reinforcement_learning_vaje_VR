%% Create frozen lake env
vpisna_stevilka = 649051670;
rng(vpisna_stevilka) 
n = 8;

lake = -1*ones(n,n);

for i=1:n
    for j=1:n
        if (rand() < 0.25)
            lake(i,j) = -n;
        end
    end
end
lake(1,1) = -1;
lake(1,2) = -1;
lake(2,1) = -1;
lake(2,2) = -1;
lake(n-1,n-1) = -1;
lake(n,n-1) = -1;
lake(n-1,n) = -1;
lake(n,n) = n;

% Render environment
disp(lake)

fh = figure;
imagesc(lake);
colormap(winter);

for i=1:n
    for j=1:n
        
        if (i==1) && (j == 1)
            text(1,1,{'1','START'},'HorizontalAlignment','center');
        elseif (i==n) && (j==n)
            text(n,n,{num2str(n*n),'GOAL'},'HorizontalAlignment','center')
        else
            text(j,i,num2str(i+n*(j-1)),'HorizontalAlignment','center')
        end
    end
end

axis off

%%
%Vaša koda

% Initialize Q-table
Q = zeros(n*n, 5);

% Set hyperparameters
alpha = 0.1;  % learning rate
gamma = 0.9;  % discount factor
epsilon = 0.5;  % exploration rate
trial_length = 0;
% Run Q-learning algorithm
num_episodes = 10000;
for episode = 1:num_episodes
    state = 1;  % start state
    done = false;
    trial_length = 0;
    while ~done
        % Choose action using epsilon-greedy policy
        if rand() < epsilon
            action = randi(5);  % explore
        else
            [~, action] = max(Q(state, :));  % exploit
        end
        
        % Perform action and observe next state and reward
        next_state = transition(state, action,n);
        reward = reward_function(next_state, lake, n);
        
        % Update Q-table
        Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state, :)) - Q(state, action));
        
        % Update current state
        state = next_state;
        
        % Check if episode is done
        if state == n*n
            done = true;
        end
        if trial_length > 50
            done = true;
        end
        trial_length = trial_length+1;
    end
end

%%
% Vizualizacija rešitve
indexQ = int32([(1:(n*n))]');
visQ = table(indexQ,Q)

num_steps = visualization_Q5(Q, lake);
num_steps = visualization_Q_arrows5(Q, lake);

