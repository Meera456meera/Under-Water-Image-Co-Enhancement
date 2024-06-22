import time
import numpy as np

def CO(cheetahs,objective_function,lb,ub,n_iterations):
    n_cheetahs,bounds = cheetahs.shape[0],cheetahs.shape[1]
    # Initialize the best position and score
    best_position = None
    best_score = -np.inf
    # The probability of choosing searching or sitting-and-waiting strategy
    alpha = 0.5
    # The probability of choosing attacking strategy
    beta = 0.5
    # The probability of leaving the prey and going back home
    delta = 0.5
    Convergence_curve = np.zeros((n_iterations))
    ct = time.time()
    # Run the optimization loop
    for i in range(n_iterations):
        # Evaluate the objective function for each cheetah
        scores = np.array([objective_function(cheetah) for cheetah in cheetahs])
        if np.max(scores) > best_score:
            best_score = np.max(scores)
            best_position = cheetahs[np.argmax(scores)]
            # Create a new array to store the updated positions
            new_cheetahs = np.zeros_like(cheetahs)
            # Loop through each cheetah
            for j in range(n_cheetahs):
                # Generate a random number between 0 and 1
                r = np.random.rand()
                # If r < alpha, choose searching or sitting-and-waiting strategy
                if r < alpha:
                    # Generate another random number between 0 and 1
                    s = np.random.rand()
                    # If s < 0.5, choose searching strategy
                    if s < 0.5:
                        # Choose a random function from a pool of functions
                        f = np.random.choice([np.sin, np.cos, np.tan])
                        # Update the position by following a leader using the function
                        leader = cheetahs[np.argmax(scores)]
                        new_cheetahs[j] = leader + f((i + 1) / n_iterations) * (leader - cheetahs[j])
                    # Else, choose sitting-and-waiting strategy
                    else:
                        # Update the position by adding a random perturbation
                        new_cheetahs[j] = cheetahs[j] +np.random.uniform(-0.01, 0.01, size=len(bounds))
                # Else if r < alpha + beta, choose attacking strategy
                elif r < alpha + beta:
                    # Update the position by moving towards the best position with a random factor
                    new_cheetahs[j] = cheetahs[j] + np.random.rand() * (best_position - cheetahs[j])
        # Else, choose leaving-the-prey-and-going-back-home strategy
        else:
            # Update the position by moving back to the initial position
            # with some random perturbation
            new_cheetahs[j] = cheetahs[0] + np.random.uniform(-delta, delta, size=len(bounds))
            new_cheetahs[j] = np.clip(new_cheetahs[j], lb, ub)
        cheetahs = new_cheetahs
        cValue = best_position[0]

        Convergence_curve[i] = cValue
    ct = time.time() - ct
    return best_position,Convergence_curve,best_score,ct