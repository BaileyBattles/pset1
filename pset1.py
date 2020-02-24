#GLOBALS
L = 5
H = 6
gamma = 0.9
p_e = 0.01
epsilon = 0.01

class Action:
    def __init__(self, move):
        if move != 'left' and move != 'right' and move != 'up' and move != 'down' and move != 'none':
            print("MOVE MUST BE A VALID MOVE")
        self.move = move

class StateSpace:
    """
    State is a H by L grid where a 0 represents an open space and a 1 represents an obstacle
    ObstacleList should be a list of tuples where each tuple is (row, col) of the obstacle
    RewardList should be a three tuple (row, col, rewardValue)
    """
    def __init__(self, L, H, obstacleList = [], rewardList = []):
        self.numRows = H
        self.numCols = L
        self.stateSpace = [[0 for x in range(self.numCols)] for x in range(self.numRows)]
        for obstacle in obstacleList:
            self.stateSpace[obstacle[0]][obstacle[1]] = -1
        self.rewardMap = {}
        for reward in rewardList:
            self.rewardMap[(reward[0], reward[1])] = reward[2]
    def validPosition(self, row, col):
        if row < 0 or col < 0 or row >= self.numRows or col >= self.numCols:
            return False
        #2a
        if self.stateSpace[row][col] == -1:
            return False
        return True
    
    #2b
    def getReward(self, row, col):
        if (row, col) in self.rewardMap:
            return self.rewardMap[(row, col)]
        else:
            return 0
    
    def getMoves(self, row, col):
        moves = [(row, col)]
        if stateSpace.validPosition(row - 1, col):
            moves.append((row - 1, col))
        
        if stateSpace.validPosition(row + 1, col):
            moves.append((row + 1, col))
        
        if stateSpace.validPosition(row, col - 1):
            moves.append((row, col - 1))
        
        if stateSpace.validPosition(row, col + 1):
            moves.append((row, col + 1))
        return moves
    
    def getActions(self, row, col):
        actions = [Action('none')]
        if stateSpace.validPosition(row - 1, col):
            actions.append(Action('up'))
        
        if stateSpace.validPosition(row + 1, col):
            actions.append(Action('down'))
        
        if stateSpace.validPosition(row, col - 1):
            actions.append(Action('left'))
        
        if stateSpace.validPosition(row, col + 1):
            actions.append(Action('right'))
        return actions
    
    def nextIntendedPosition(self, row, col, action):
        if action.move == 'left':
            return (row, col - 1)
        elif action.move == 'right':
            return (row, col + 1)
        elif action.move == 'none':
            return (row, col)
        elif action.move == 'up':
            return (row - 1, col)
        elif action.move == 'down':
            return (row + 1, col)
        else:
            print("Not a valid move")
    
    def probabilityNewState(self, s, a, s0):
        if not stateSpace.validPosition(s[0], s[1]):
            return 0

        if (s[0], s[1]) not in stateSpace.getMoves(s0[0], s0[1]):
            return 0

        intendedRow, intendedCol = stateSpace.nextIntendedPosition(s0[0], s0[1], a)
        if s[0] == intendedRow and s[1] == intendedCol:
            if a.move == 'none':
                return 1.
            else:
                return 1. - p_e + p_e/4

        #If next state is a position robot was not intending to go
        elif s[0] == s0[0] and s[1] == s0[1]:
            return 0
        else:
            return p_e
    
    def valueFunction(self):
        value_fn = [[0 for x in range(self.numCols)] for x in range(self.numRows)]
        stable = False
        currPolicy = Policy([[Action('none') for x in range(self.numCols)] for x in range(self.numRows)])
        while not stable:
            delta = 0
            new_value_fn = [[0 for x in range(self.numCols)] for x in range(self.numRows)]
            for row in range(self.numRows):
                for col in range(self.numCols):
                    actions = self.getActions(row, col)
                    moves = self.getMoves(row, col)
                    bestValue = 0
                    for action in actions:
                        for move in moves:
                            rewardOfMove = self.getReward(move[0], move[1]) + gamma * value_fn[move[0]][move[1]]
                            newValue = self.probabilityNewState(move, action, (row,col)) * rewardOfMove
                            if newValue > bestValue:
                                bestValue = newValue
                                currPolicy.policyMatrix[row][col] = action
                    new_value_fn[row][col] = bestValue
                    delta = max(delta, abs(new_value_fn[row][col] - value_fn[row][col]))

            value_fn = new_value_fn
            if delta < epsilon:
                stable = True
        return (new_value_fn, currPolicy)
    

class Policy:
    def __init__(self, policyMatrix):
        # PolicyFunction should take two arguments, stateSpace and state
        self.policyMatrix = policyMatrix
    
    def displayPolicy(self, stateSpace):
        display = [["" for x in range(stateSpace.numCols)] for x in range(stateSpace.numRows)]
        for row in range(stateSpace.numRows):
            for col in range(stateSpace.numCols):
                if not stateSpace.validPosition(row, col):
                    display[row][col] = 'X'
                else:
                    display[row][col] = self.policyMatrix[row][col].move
        for row in display:
            print(row)  
    
    def evaluatePolicy(self, stateSpace):
        policy_values = [[0 for x in range(len(self.policyMatrix[0]))] for x in range(len(self.policyMatrix))]
        stable = False
        while not stable:
            delta = 0
            new_policy_values = [[0 for x in range(len(self.policyMatrix[0]))] for x in range(len(self.policyMatrix))]
            
            for row in range(len(self.policyMatrix)):
                for col in range(len(self.policyMatrix[row])):
                    action = self.policyMatrix[row][col]
                    for move in stateSpace.getMoves(row, col):
                        reward = stateSpace.getReward(move[0], move[1]) + gamma * policy_values[move[0]][move[1]]
                        new_policy_values[row][col] += stateSpace.probabilityNewState(move, action, (row, col)) * reward
                    delta = max(delta, abs(new_policy_values[row][col] - policy_values[row][col]))
            
            policy_values = new_policy_values
            if delta < epsilon:
                stable = True
        return policy_values
    
    def evaluatePolicyForState(policy, stateSpace, state):
        return policy.evaluatePolicyForStateHelper(value_fn, stateSpace, state, 1, 0)

    def getPathAndRewardFrom(self, row, col, stateSpace):
        """
        Retruns the path and the reward from the row and col
        """
        trajectory = [['.' for x in range(stateSpace.numCols)] for x in range(stateSpace.numRows)]
        if self.policyMatrix[row][col].move == 'none':
            return stateSpace.getReward(row, col)
        total_reward, iteration, expected_reward = 0, 0, 0
        path = []
        delta = epsilon
        while True:
            trajectory[row][col] = '*'
            delta = pow(gamma, iteration) * stateSpace.getReward(row, col)
            path.append(policy.policyMatrix[row][col].move)
            total_reward += delta
            nextRow, nextCol = stateSpace.nextIntendedPosition(row, col, policy.policyMatrix[row][col])
            expected_reward += delta * stateSpace.probabilityNewState((nextRow, nextCol), policy.policyMatrix[row][col], (row, col))
            if self.policyMatrix[row][col].move == 'none':
                break
            row, col = stateSpace.nextIntendedPosition(row, col, policy.policyMatrix[row][col])
            iteration += 1
        return path, trajectory, total_reward, expected_reward

def createOptimalPolicy(stateSpace):
    #initialize random policy
    policy = Policy([[Action('left') for x in range(stateSpace.numCols)] for x in range(stateSpace.numRows)])
    stable = False
    while not stable:
        newPolicy = oneStepPolicyMaker(policy, stateSpace)
        allSame = True
        for row in range(len(newPolicy.policyMatrix)):
            for col in range(len(newPolicy.policyMatrix[row])):
                if newPolicy.policyMatrix[row][col].move != policy.policyMatrix[row][col].move:
                    allSame = False
        stable = allSame
        policy = newPolicy
    return policy


def oneStepPolicyMaker(policy, stateSpace):
    value_fn = policy.evaluatePolicy(stateSpace)
    new_policy = Policy([[Action('none') for x in range(stateSpace.numCols)] for x in range(stateSpace.numRows)])
    for row in range(stateSpace.numRows):
        for col in range(stateSpace.numCols):
            new_policy.policyMatrix[row][col] = bellman_backup(policy, stateSpace, (row, col), value_fn)
    return new_policy

def bellman_backup(policy, stateSpace, state, value_fn):
    """
    Returns the best action and associated expected value of that action
    """
    actions = stateSpace.getActions(state[0], state[1])
    possibilities = []
    for action in actions:
        for move in stateSpace.getMoves(state[0], state[1]):
            value = stateSpace.probabilityNewState(move, action, state) 
            value *= stateSpace.getReward(move[0], move[1]) + gamma * value_fn[move[0]][move[1]]
            possibilities.append((value, action))
    maximum = max(possibilities, key=lambda x: x[0])
    for i in range(len(possibilities)):
        if possibilities[i] == maximum:
            return possibilities[i][1]

rewards = [(3, 2, 1), (5, 2, 10), (0, 4, -100), (1, 4, -100), (2, 4, -100), (3, 4, -100), (4, 4, -100), (5, 4, -100)]
stateSpace = StateSpace(L, H, [(2,1), (2,2), (4,1), (4,2)], rewards)
value_fn, policy = stateSpace.valueFunction()
policy.displayPolicy(stateSpace)