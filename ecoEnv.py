import random
import numpy as np
import ecoUtils
import math

class bid:
    def __init__(self):
        self.buy_metal = 0
        self.buy_wood = 0
        self.buy_food = 0
        self.buy_pick = 0
        self.buy_work = 0
        self.sell_metal = 0
        self.sell_wood = 0
        self.sell_food = 0
        self.sell_pick = 0
        self.sell_work = 0

class entity:
    def __init__(self):
        self.productivity = 1
        self.metal = 10
        self.wood = 10
        self.food = 10
        self.pick = 10
        self.work = 10
        self.rewards_this_turn = 0
        self.bids = []

    def makeBid(self, sell_metal, sell_wood, sell_food, sell_pick, sell_work, buy_metal, buy_wood, buy_food, buy_pick, buy_work):
        if sell_metal != 0 or sell_wood != 0 or sell_food != 0 or sell_pick != 0 or sell_work != 0 or buy_metal != 0 or buy_wood != 0 or \
                buy_food != 0 or buy_pick != 0 or buy_work != 0:

            myBid = bid()

            myBid.sell_metal = min(sell_metal, self.metal)
            self.metal = self.metal - myBid.sell_metal

            myBid.sell_wood = min(sell_wood, self.wood)
            self.wood = self.wood - myBid.sell_wood

            myBid.sell_food = min(sell_food, self.food)
            self.food = self.food - myBid.sell_food

            myBid.sell_pick = min(sell_pick, self.pick)
            self.pick = self.pick - myBid.sell_pick

            myBid.sell_work = min(sell_work, self.work)
            self.work = self.work - myBid.sell_work

            myBid.buy_metal = buy_metal
            myBid.buy_wood = buy_wood
            myBid.buy_food = buy_food
            myBid.buy_pick = buy_pick
            myBid.buy_work = buy_work

            self.bids.append(myBid)

    def returnBids(self):
        for bid_index in range(len(self.bids)-1,0,-1):
            bid = self.bids[bid_index]
            self.metal = self.metal + bid.sell_metal
            self.wood = self.wood + bid.sell_wood
            self.food = self.food + bid.sell_food
            self.pick = self.pick + bid.sell_pick
            del self.bids[bid_index]

class player(entity):

    def __init__(self):
        self.alive = True
        self.productivity = 1
        self.metal = 10
        self.wood = 10
        self.food = 10
        self.pick = 10
        self.work = 1
        self.rewards_this_turn = 0
        self.bids = []

    def consume(self):
        if self.alive:
            if self.food > 0:
                self.food = self.food - 1
                self.work = 1
                self.rewards_this_turn = self.rewards_this_turn + 1
            else:
                self.alive = False

class mine(entity):

    def __init__(self):
        self.productivity = 1
        self.metal = 10
        self.wood = 10
        self.food = 10
        self.pick = 10
        self.work = 5
        self.rewards_this_turn = 0
        self.bids = []

    def produce(self):
        while self.work > 0:
            self.work = self.work - 1
            if self.pick > 0:
                self.pick = self.pick - 1
                self.metal = self.metal + self.productivity * 5
                self.rewards_this_turn = self.rewards_this_turn + self.productivity * 5
            else:
                self.pick = self.pick - 1
                self.metal = self.metal + self.productivity
                self.rewards_this_turn = self.rewards_this_turn + self.productivity

class forest(entity):

    def __init__(self):
        self.productivity = 1
        self.metal = 10
        self.wood = 10
        self.food = 10
        self.pick = 10
        self.work = 5
        self.rewards_this_turn = 0
        self.bids = []

    def produce(self):
        while self.work > 0:
            self.work = self.work - 1
            if self.pick > 0:
                self.pick = self.pick - 1
                self.wood = self.wood + self.productivity * 5
                self.rewards_this_turn = self.rewards_this_turn + self.productivity * 5
            else:
                self.wood = self.wood + self.productivity
                self.rewards_this_turn = self.rewards_this_turn + self.productivity

class farm(entity):

    def __init__(self):
        self.productivity = 1
        self.metal = 10
        self.wood = 10
        self.food = 10
        self.pick = 10
        self.work = 5
        self.rewards_this_turn = 0
        self.bids = []

    def produce(self):
        while self.work > 0:
            self.work = self.work - 1
            if self.pick > 0:
                self.pick = self.pick - 1
                self.food = self.food + self.productivity * 10
                self.rewards_this_turn = self.rewards_this_turn + self.productivity * 10
            else:
                self.food = self.food + self.productivity
                self.rewards_this_turn = self.rewards_this_turn + self.productivity

class factory(entity):

    def __init__(self):
        self.productivity = 1
        self.metal = 10
        self.wood = 10
        self.food = 10
        self.pick = 10
        self.work = 5
        self.rewards_this_turn = 0
        self.bids = []

    def produce(self):
        while self.work > 0 and self.metal > 1 and self.wood > 1:
            self.work = self.work - 1
            self.metal = self.metal - 1
            self.wood = self.wood - 1
            self.pick = self.pick + 1
            self.rewards_this_turn = self.rewards_this_turn + 1

#######################
#
# Environment Steps
#
#######################

# Entities are initialized
# Entities look around - the AI asks for the state
# Bids are established - the AI provides bid orders
# Bids are matched and executed were possible
# Entities produce as much as they can given inventories
# Work Inventories are erased
# Bids are returned (except work)
# Players consume and get work in their inventory
# Game metrics are calculated and plotted
# Entity accumulated rewards set to zero

class market:
    def __init__(self,experiment):
        self.entities = []
        self.production = 0
        self.frame = 0
        self.experiment = experiment
        self.reset(playerCt=60,mineCt=10,forestCt=15,farmCt=20,factoryCt=5)

    #############################################
    #### External Functions
    #############################################

    def entityCount(self):
        return len(self.entities)

    def reset(self,playerCt,mineCt,forestCt,farmCt,factoryCt):
        self.entities = []
        for index in range(playerCt):
            x = player()
            self.entities.append(x)
        for index in range(mineCt):
            x = mine()
            self.entities.append(x)
        for index in range(forestCt):
            x = forest()
            self.entities.append(x)
        for index in range(farmCt):
            x = farm()
            self.entities.append(x)
        for index in range(factoryCt):
            x = factory()
            self.entities.append(x)
        random.shuffle(self.entities)

    def getStateExternal(self,width,entityID):
        # Width indicates the size of the numpy array to return
        # entityID is the perspective from which to get the state

        state = self.getState(entityID)
        state = state[:width]
        missingEntries = max(0, width - len(state))

        for count in range(missingEntries):
            state.append(0.0)

        state = np.array(state,dtype='float')

        return state

    def takeActionsExternal(self,actionList):

        # ActionList expected to be tuple
        #   First value is the EntityID
        #   Second is integer between 0 and 19,682
        # Integers are converted into buy/sell actions
        # This conversion ensures that an agent can't make a bid to buy and sell the same item

        for action in actionList:

            entID = action[0]
            thisEntity = self.entities[entID]

            metalCode = math.floor(action[1] / 2187)
            remainder = action[1] - metalCode * 2187
            woodCode = math.floor(remainder / 243)
            remainder = remainder - woodCode * 243
            foodCode = math.floor(remainder / 27)
            remainder = remainder - foodCode * 27
            pickCode = math.floor(remainder / 3)
            remainder = remainder - pickCode * 3
            workCode = remainder

            metalCode = metalCode - 4
            woodCode = woodCode -4
            foodCode = foodCode - 4
            pickCode = pickCode - 4
            workCode = workCode - 1

            if metalCode < 0:
                sell_metal = -1 * metalCode
                buy_metal = 0
            else:
                sell_metal = 0
                buy_metal = metalCode


            if woodCode < 0:
                sell_wood = -1 * woodCode
                buy_wood = 0
            else:
                sell_wood = 0
                buy_wood = woodCode


            if foodCode < 0:
                sell_food = -1 * foodCode
                buy_food = 0
            else:
                sell_food = 0
                buy_food = foodCode


            if pickCode < 0:
                sell_pick = -1 * pickCode
                buy_pick = 0
            else:
                sell_pick = 0
                buy_pick = pickCode

            if workCode < 0:
                sell_work = -1 * workCode
                buy_work = 0
            else:
                sell_work = 0
                buy_work = workCode

            thisEntity.makeBid(sell_metal, sell_wood, sell_food, sell_pick, sell_work, buy_metal, buy_wood, buy_food, buy_pick, buy_work)

        self.trade()
        self.produce()
        self.eraseWork()
        self.returnBids()
        self.playersConsume()
        self.calcMetrics()

        # Report reward by entity
        rewards = []
        for action in actionList:
            rewards.append(self.entities[action[0]].rewards_this_turn)

        # Set entity rewards to zero
        for ent in self.entities:
            ent.rewards_this_turn = 0

        # Terminal if too many players have died
        terminal = self.isTerminal()

        return rewards, terminal

    #############################################
    #### Internal-only Functions
    #############################################

    def isTerminal(self):

        alive = 0
        dead = 0
        for ent in self.entities:
            if isinstance(ent,player):
                if ent.alive:
                    alive = alive + 1
                else:
                    dead = dead + 1

        total  = alive + dead
        if total > 0:
            if alive / total < 0.25:
                return True
            else:
                return False
        return True

    def trade(self):
        for ent1 in self.entities:
            for ent2 in self.entities:
                if ent1 is ent2:
                    pass
                else:
                    for bid1_index in range(len(ent1.bids)-1, 0, -1):
                        for bid2_index in range(len(ent2.bids)-1, 0, -1):
                            bid1 = ent1.bids[bid1_index]
                            bid2 = ent2.bids[bid2_index]

                            if self.goodDeal(bid1,bid2):

                                ent1.metal = ent1.metal + bid2.sell_metal
                                ent1.wood = ent1.wood + bid2.sell_wood
                                ent1.food = ent1.food + bid2.sell_food
                                ent1.pick = ent1.pick + bid2.sell_pick
                                ent1.work = ent1.work + bid2.sell_work

                                ent2.metal = ent2.metal + bid1.sell_metal
                                ent2.wood = ent2.wood + bid1.sell_wood
                                ent2.food = ent2.food + bid1.sell_food
                                ent2.pick = ent2.pick + bid1.sell_pick
                                ent2.work = ent2.work + bid1.sell_wood

                                del ent1.bids[bid1_index]
                                del ent2.bids[bid2_index]

    def returnBids(self):
        for ent in self.entities:
            ent.returnBids()

    def getState(self,entityID):
        state = []
        thisEntity = self.entities[entityID]

        # Add entity type to state vector
        if isinstance(thisEntity,player):
            state.append(1)
        elif isinstance(thisEntity,mine):
            state.append(2)
        elif isinstance(thisEntity,forest):
            state.append(3)
        elif isinstance(thisEntity,farm):
            state.append(4)
        elif isinstance(thisEntity,factory):
            state.append(5)

        # Add entity commodities to state vector
        state.append(self.entities[entityID].metal)
        state.append(self.entities[entityID].wood)
        state.append(self.entities[entityID].food)
        state.append(self.entities[entityID].pick)
        state.append(self.entities[entityID].work)

        # Add other entity commodities to state vector

        for ent_index in range(len(self.entities)):

            if ent_index != entityID:
                otherEntity = self.entities[ent_index]
                if isinstance(thisEntity,player):
                    state.append(1)
                elif isinstance(thisEntity,mine):
                    state.append(2)
                elif isinstance(thisEntity,forest):
                    state.append(3)
                elif isinstance(thisEntity,farm):
                    state.append(4)
                elif isinstance(thisEntity,factory):
                    state.append(5)

                state.append(self.entities[ent_index].metal)
                state.append(self.entities[ent_index].wood)
                state.append(self.entities[ent_index].food)
                state.append(self.entities[ent_index].pick)
                state.append(self.entities[ent_index].work)

        return state

    def produce(self):
        for ent in self.entities:
            if not isinstance(ent,player):
                ent.produce()
        for ent in self.entities:
            self.production = self.production + ent.rewards_this_turn

    def eraseWork(self):
        for ent in self.entities:
            ent.work = 0

    def playersConsume(self):
        for ent in self.entities:
            if isinstance(ent,player):
                ent.consume()

    def goodDeal(self,bid1,bid2):
        isGood = (bid1.sell_metal >= bid2.buy_metal) and (bid2.sell_metal >= bid1.buy_metal)
        isGood = isGood and (bid1.sell_wood >= bid2.buy_wood) and (bid2.sell_wood >= bid1.buy_wood)
        isGood = isGood and (bid1.sell_food >= bid2.buy_food) and (bid2.sell_food >= bid1.buy_food)
        isGood = isGood and (bid1.sell_pick >= bid2.buy_pick) and (bid2.sell_pick >= bid1.buy_pick)
        isGood = isGood and (bid1.sell_work >= bid2.buy_work) and (bid2.sell_work >= bid1.buy_work)

        return isGood

    def calcMetrics(self):

        self.frame = self.frame + 1

        totalProduction = 0

        for ent in self.entities:
            totalProduction = totalProduction + ent.rewards_this_turn

        ecoUtils.writeMetrics(totalProduction,self.frame,self.experiment)

        print(f'Frame: {self.frame}   Production: {totalProduction}')

        # Total Production This Round
        # Average Production by Entity Type
        # Average Assets by Asset Type and Entity Type
        # Min Assets by Asset Type and Entity Type
        # Max Assets by Asset Type and Entity Type









