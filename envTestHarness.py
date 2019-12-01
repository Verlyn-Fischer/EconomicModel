import ecoEnv
import random

experiment = 'test2'
market = ecoEnv.market(experiment)
entityCount = market.entityCount()

for step in range(10):

    for entID in range(entityCount):

        currentState = market.getStateExternal(width=1000,entityID=entID)

    actionList = []

    # sell_metal, sell_wood, sell_food, sell_pick, sell_work, buy_metal, buy_wood, buy_food, buy_pick, buy_work
    for entID in range(entityCount):
        r = random.randint(0,19682)

        action = (entID,r)

        actionList.append(action)

    rewards, terminal = market.takeActionsExternal(actionList)

