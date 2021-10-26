WALL  = 3
EMPTY = 0
UNOBSERVED = 6
REWARD = 2 
AGENT = 5

# For multicolor walls
GREEN_WALL  = 7
PRUPLE_WALL  = 8


# For rotational agent
AGENT_UP = 5
AGENT_RIGHT = 1
AGENT_DOWN = 4
AGENT_LEFT = 9

def is_agent(a):
	return a == AGENT_UP or a == AGENT_RIGHT or a == AGENT_DOWN or a == AGENT_LEFT

def is_wall(a):
	return a == WALL or a == GREEN_WALL or a == PRUPLE_WALL

semantics_map = {
	WALL: "wall",
	EMPTY: "empty",
	UNOBSERVED: "unobserved",
	REWARD:"reward",
	AGENT: "agent"
}

inverse_semantics_map = {v: k for k, v in semantics_map.items()}