# Defining DQN-HER Algorithm
import copy

n = 2
grid = grid_nxn(n)

state_shape = grid.returnState().shape[0]
action_shape = 4



def dqn_her(n_episodes=100000, max_t=int(3*n), eps_start=1.0, eps_end=0.1, eps_decay=0.99995):
    print(pow(eps_decay,n_episodes))

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window= deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    hindsight_goals = []
   
    #Check if agent learns to solve a cube that is one move away from goal state
    
    
    for i_episode in range(1, n_episodes+1):
        
        grid = grid_nxn(n)
        fin_goal = grid.goal
        state = grid.returnState()
        score = 0
        done = False
        h_flag = 0
        
        
        traj_val = []
         
        # Run for one trajectory
        for t in range(max_t):
            
            #We store all we need for each trajectory
            
            #print(grid.returnState())
            
            #Choosing an action
            action = agent.act(np.concatenate((state,fin_goal)), eps)

            #Executing that action
            grid.move(action)
            
            #Next state
            next_state = grid.returnState()
            
            #Modified reward system
            reward = grid.checkReward()
        
            
            #Checking if the episode ended
            if grid.checkDone():
                done = True
             
            #print(state)
            traj_val.append([state,action,reward,next_state,done])
            #print(traj_val)
            #agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break 
                
        # Once the trajectory is done, append the final state that the trajectory reached to the state and push it to experience replay
        psuedo_goal = next_state
        
        
                
        if (not(psuedo_goal == fin_goal).all()):
            flag = 0
        #if state != fin_goal:   
            if len(hindsight_goals) == 0:
                print('yes1')
                hindsight_goals.append(psuedo_goal)
            for hind_goal in hindsight_goals:
                if ((state == hind_goal).all()):
                    flag = 1
            if flag == 0:
                hindsight_goals.append(psuedo_goal)
        
        for sublist in traj_val:
            new_state = np.concatenate((sublist[0],psuedo_goal))
            new_next_state = np.concatenate((sublist[3],psuedo_goal))
            agent.step(new_state, sublist[1], sublist[2], new_next_state, sublist[4])
            
        #Working on the hindsight learning
        #print(len(hindsight_goals))
        for hind_goal in hindsight_goals:
            for sublist in traj_val:
                #Altering the input state structure
                new_state = np.concatenate((sublist[0],hind_goal))
                
                #Altering the reward
                if ((sublist[3] == hind_goal).all):
                    new_reward = 1 
                    
                #Altering the next state structure
                new_next_state = np.concatenate((sublist[3],hind_goal))
                agent.step(new_state, sublist[1], reward, new_next_state, sublist[4])
                
        
        #Training (Refer to HER algorithm)
        for _ in range(max_t):
            agent.train_call()
            
            
        
        
            
        scores_window.append(score)                       # save most recent score
        scores_window_printing.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)                 # decrease epsilon
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score), end="")        
        if i_episode % 100 == 0: 
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            scores.append(np.mean(scores_window))
        if np.mean(scores_window)>=100 - 2.5*n:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    return [np.array(scores),i_episode-100]

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
agent = learning_agents.Agent_DQNHER(state_size=state_shape,action_size = action_shape,seed = 0)
scores_her, terminal_ep_her = dqn_her()