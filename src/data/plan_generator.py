from keras.utils import Sequence
import numpy as np

from src.constants import RANDOM_SEED


class PlanGenerator(Sequence):

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        shuffle=True,
        truncate=True,
        zero_padding=True,
    ):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.shuffle = shuffle
        self.truncate = truncate
        self.zero_padding = zero_padding

    def __getitem__(self, index):
        # FIXME: implement the __getitem__ method
        pass

    def __len__(self):
        return len(self.plans) // self.batch_size

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.plans)

    def get_actions(
        self,
        actions,
        perc,
    ):
        MIN_NUMBER_ACTIONS = 1
        # Compute the number of the action sequence after perc sampling
        num_actions = max(np.ceil(len(actions) * perc).astype(int), MIN_NUMBER_ACTIONS)
        # Get the indexes of the actions to sample
        indexes = np.arange(len(actions))
        # Sample the indexes and resort them
        sampled_indexes = np.sort(np.random.choice(indexes, num_actions, replace=False))

        if self.truncate == True:
            # Truncate the indexes sequence to max_dim
            sampled_indexes = sampled_indexes[: self.max_dim]

        action_ids = [
            self.dizionario[a.name] for a in np.take(actions, sampled_indexes)
        ]

        if self.zero_padding == True and len(action_ids) < self.max_dim:
            # Pad the sequence with 0 values until max_dim
            action_ids += [0] * (self.max_dim - len(action_ids))

        return action_ids

    def get_goal_mask(self, goal):
        # Get the subgoals one hot encoding of a specific goal from the dictionary of goals
        subgoals_mask = [self.dizionario_goal[subgoal] for subgoal in goal]

        # Compute the goal mask as the sum of the subgoals one hot encoding
        goal_mask = np.array(subgoals_mask).sum(axis=0)

        return goal_mask


class PlanGeneratorPerc(PlanGenerator):

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        perc,
        shuffle=True,
        truncate=True,
        zero_padding=True,
    ):
        super().__init__(
            plans,
            dizionario,
            dizionario_goal,
            batch_size,
            max_dim,
            shuffle,
            truncate,
            zero_padding,
        )
        self.perc = perc

    def __getitem__(self, index):
        batch = self.plans[index * self.batch_size : (index + 1) * self.batch_size]

        # Get the actions (padded or truncated) and the goal mask for each plan in the batch
        batch_actions = [self.get_actions(plan.actions, self.perc) for plan in batch]
        batch_goal_masks = [self.get_goal_mask(plan.goals) for plan in batch]

        # Convert the lists of actions and goal masks into numpy arrays X and Y
        X = np.array(batch_actions)
        Y = np.array(batch_goal_masks)

        return X, Y


class PlanGeneratorMultiPerc(PlanGenerator):

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        min_perc,
        max_perc,
        shuffle=True,
        truncate=True,
        zero_padding=True,
    ):
        super().__init__(
            plans,
            dizionario,
            dizionario_goal,
            batch_size,
            max_dim,
            shuffle,
            truncate,
            zero_padding,
        )
        self.min_perc = min_perc
        self.perc = max_perc

    @staticmethod
    def get_seed_from_plan_name(plan):
        seed = plan.plan_name.rsplit("-p", 1)[1]
        seed = seed.split(",", 1)[0]
        if not seed.isdigit():
            seed = seed.split(".", 1)[0]
        return int(seed)

    def get_random_perc(self, seed):
        np.random.seed(seed)
        return np.random.uniform(self.min_perc, self.perc)

    def get_random_perc_from_plan_name(self, plan):
        seed = self.get_seed_from_plan_name(plan)
        return self.get_random_perc(seed)

    def __getitem__(self, index):
        batch = self.plans[index * self.batch_size : (index + 1) * self.batch_size]

        # Get the actions (padded or truncated) and the goal mask for each plan in the batch
        batch_actions = [
            self.get_actions(plan.actions, self.get_random_perc_from_plan_name(plan))
            for plan in batch
        ]

        batch_goal_masks = [self.get_goal_mask(plan.goals) for plan in batch]

        # Convert the lists of actions and goal masks into numpy arrays X and Y
        X = np.array(batch_actions)
        Y = np.array(batch_goal_masks)

        return X, Y


np.random.seed(RANDOM_SEED)
class PlanGeneratorPerc(PlanGenerator):

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        perc,
        shuffle=True,
        truncate=True,
        zero_padding=True,
    ):
        super().__init__(
            plans,
            dizionario,
            dizionario_goal,
            batch_size,
            max_dim,
            shuffle,
            truncate,
            zero_padding,
        )
        self.perc = perc

    def __getitem__(self, index):
        batch = self.plans[index * self.batch_size : (index + 1) * self.batch_size]

        # Get the actions (padded or truncated) and the goal mask for each plan in the batch
        batch_actions = [self.get_actions(plan.actions, self.perc) for plan in batch]
        batch_goal_masks = [self.get_goal_mask(plan.goals) for plan in batch]

        # Convert the lists of actions and goal masks into numpy arrays X and Y
        X = np.array(batch_actions)
        Y = np.array(batch_goal_masks)

        return X, Y


class PlanGeneratorOnline(PlanGenerator):

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        shuffle=True,
        truncate=True,
        zero_padding=True,
    ):
        super().__init__(
            plans,
            dizionario,
            dizionario_goal,
            batch_size,
            max_dim,
            shuffle,
            truncate,
            zero_padding,
        )


    def get_actions(
        self,
        actions
    ):
        
        if self.truncate == True:
            # Truncate the indexes sequence to max_dim
            actions = actions[: self.max_dim]

        action_ids = [
            self.dizionario[a.name] for a in actions
        ]

        if self.zero_padding == True and len(action_ids) < self.max_dim:
            # Pad the sequence with 0 values until max_dim
            action_ids += [0] * (self.max_dim - len(action_ids))

        return action_ids

    def get_goal_mask(self, goal):
        # Get the subgoals one hot encoding of a specific goal from the dictionary of goals
        subgoals = []
        for n in range(self.max_dim):
            if n<len(goal):
                subgoals_mask = [self.dizionario_goal[subgoal] for subgoal in goal[n] if subgoal in self.dizionario_goal]
            else:
                subgoals_mask = [self.dizionario_goal[subgoal] for subgoal in goal[len(goal)-1] if subgoal in self.dizionario_goal]
                
            # Compute the goal mask as the sum of the subgoals one hot encoding
            goal_mask = np.array(subgoals_mask).sum(axis=0)
            # Cast all elements of goal_mask to int
            try:
                goal_mask = list(map(int, goal_mask.tolist()))
            except:
                break
            subgoals.append(goal_mask)

        return subgoals
    
    @staticmethod
    def get_seed_from_plan_name(plan):
        seed = plan.plan_name.rsplit("-p", 1)[1]
        seed = seed.split(",", 1)[0]
        if not seed.isdigit():
            seed = seed.split(".", 1)[0]
        return int(seed)

    def get_random_perc(self, seed):
        np.random.seed(seed)
        return np.random.uniform(self.min_perc, self.perc)

    def get_random_perc_from_plan_name(self, plan):
        seed = self.get_seed_from_plan_name(plan)
        return self.get_random_perc(seed)

    def __getitem__(self, index):
        batch = self.plans[index * self.batch_size : (index + 1) * self.batch_size]

        # Get the actions (padded or truncated) and the goal mask for each plan in the batch
        batch_actions = [
            self.get_actions(plan.actions)
            for plan in batch
        ]

        #batch_goal_masks = [self.get_goal_mask(plan.goals) for plan in batch]

        
        batch_goal_masks = []
        for plan in batch:
            try:
                #If the count of 1 is 0
                if self.get_goal_mask(plan.goals).count(1)==0:
                    print("If "+ plan.plan_name)
                batch_goal_masks.append(self.get_goal_mask(plan.goals))
            except:
                print("Else "+ plan.plan_name)
                break
        
        
        # Convert the lists of actions and goal masks into numpy arrays X and Y
        X = batch_actions
        Y = batch_goal_masks

        return X, Y

class PlanGeneratorOnlineSimple(PlanGenerator):

    def __init__(
        self,
        plans,
        dizionario,
        dizionario_goal,
        batch_size,
        max_dim,
        shuffle=True,
        truncate=True,
        zero_padding=True,
    ):
        super().__init__(
            plans,
            dizionario,
            dizionario_goal,
            batch_size,
            max_dim,
            shuffle,
            truncate,
            zero_padding,
        )


    def get_actions(
        self,
        actions
    ):
        
        if self.truncate == True:
            # Truncate the indexes sequence to max_dim
            actions = actions[: self.max_dim]

        action_ids = [
            self.dizionario[a.name] for a in actions
        ]

        if self.zero_padding == True and len(action_ids) < self.max_dim:
            # Pad the sequence with 0 values until max_dim
            action_ids += [0] * (self.max_dim - len(action_ids))

        return action_ids

    def get_goal_mask(self, goal):
        # Get the subgoals one hot encoding of a specific goal from the dictionary of goals
        subgoals = []
        for n in range(self.max_dim):
            if n<len(goal):
                subgoals_mask = [self.dizionario_goal[subgoal] for subgoal in goal[n] if subgoal in self.dizionario_goal]
            else:
                subgoals_mask = [self.dizionario_goal[subgoal] for subgoal in goal[len(goal)-1] if subgoal in self.dizionario_goal]
                
            # Compute the goal mask as the sum of the subgoals one hot encoding
            goal_mask = np.array(subgoals_mask).sum(axis=0)
            # Cast all elements of goal_mask to int
            try:
                goal_mask = list(map(int, goal_mask.tolist()))
            except:
                break
            #Append the goal_mask to the subgoals list as the index of the ones in the goal_mask
            subgoals.append([i for i in range(len(goal_mask)) if goal_mask[i]==1])
        return subgoals
    
    @staticmethod
    def get_seed_from_plan_name(plan):
        seed = plan.plan_name.rsplit("-p", 1)[1]
        seed = seed.split(",", 1)[0]
        if not seed.isdigit():
            seed = seed.split(".", 1)[0]
        return int(seed)

    def get_random_perc(self, seed):
        np.random.seed(seed)
        return np.random.uniform(self.min_perc, self.perc)

    def get_random_perc_from_plan_name(self, plan):
        seed = self.get_seed_from_plan_name(plan)
        return self.get_random_perc(seed)

    def __getitem__(self, index):
        batch = self.plans[index * self.batch_size : (index + 1) * self.batch_size]

        # Get the actions (padded or truncated) and the goal mask for each plan in the batch
        batch_actions = [
            self.get_actions(plan.actions)
            for plan in batch
        ]

        #batch_goal_masks = [self.get_goal_mask(plan.goals) for plan in batch]

        
        batch_goal_masks = []
        for plan in batch:
            print(self.get_goal_mask(plan.goals))
            try:
                #If the count of 1 is 0
                print(self.get_goal_mask(plan.goals))
                if self.get_goal_mask(plan.goals):
                    print("If "+ plan.plan_name)
                batch_goal_masks.append(self.get_goal_mask(plan.goals))
            except:
                print("Else "+ plan.plan_name)
                break
        
        
        # Convert the lists of actions and goal masks into numpy arrays X and Y
        X = batch_actions
        Y = batch_goal_masks

        return X, Y


np.random.seed(RANDOM_SEED)
