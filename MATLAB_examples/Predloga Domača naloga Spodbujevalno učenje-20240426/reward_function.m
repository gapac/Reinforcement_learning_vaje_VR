function reward = reward_function(next_state, lake, n)
    % This function should take the next state and return the corresponding reward.
    % You need to fill in the logic based on your specific problem.
    
    %For example, if your reward is a function of the state, you might calculate the reward like this:
    reward = -1;  % default reward
    % Convert state to row and column indices
    [row, col] = ind2sub([n n], next_state);

    if lake(row, col) == n
        reward = 500;  % reward for reaching the goal
    end
    if lake(row, col) == -n
        reward = -800;  % reward for falling in the hole
    end
end