function next_state = transition(state, action, n)
    % This function should take the current state and action, and return the next state.
    % You need to fill in the logic based on your specific problem.
    
    % For example, if your state is a position on a grid and action is a direction to move,
    % you might calculate the next state like this:
    % 
    switch action
        case 4  % move up
            next_state = state - 1; 
            %if state is 1 0r 5 or 9 or 13 then next state will be same as state
            if mod(state, n) == 1
                next_state = state;
            end
        case 3  % move right
            next_state = state + n;
            if state >= (n*n)-n
                next_state = state;
            end
        case 2  % move down
            next_state = state + 1;
            if mod(state, n) == 0
                next_state = state;
            end
        case 1  % move left
            next_state = state - n;
            if state  <= n
                next_state = state;
            end
        case 5  % move right, down
            next_state = state + n + 1;
            if state >= (n*n)-n
                next_state = state;
            end
            if mod(state, n) == 0
                next_state = state;
            end
    end
    
    
end