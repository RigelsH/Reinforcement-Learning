class StateRepresentation:
    """
    Class describes the state of the envirnonmet.
    """
    def __init__(self, 
                 agent_x_coord=0, 
                 agent_y_coord=0, 
                 agent_orientation=(0, 1),
                 has_agent_grabbed_gold=False, 
                 has_agent_climbed_out=False,
                 has_agent_shot_arrow=False):
        
        self.agent_x_coord = agent_x_coord
        self.agent_y_coord = agent_y_coord
        self.agent_orientation = agent_orientation
        self.has_agent_grabbed_gold = has_agent_grabbed_gold
        self.has_agent_climbed_out = has_agent_climbed_out
        self.has_agent_shot_arrow = has_agent_shot_arrow

    def _get_bin_repr_orientation(self):
        # 2 bits max to represent four cases
        if self.agent_orientation == (0, 1): # north
            agent_orientation = 0
        elif self.agent_orientation == (1, 0): # right
            agent_orientation = 1
        elif self.agent_orientation == (0, -1): # south
            agent_orientation = 2
        elif self.agent_orientation == (-1, 0): # left
            agent_orientation = 3
        return ((str(bin(agent_orientation)))[2:]).zfill(2)
    
    def _get_bin_repr_x_coord(self):
        # 4 bits max (since we have a map with size 8)
        return ((str(bin(self.agent_x_coord)))[2:]).zfill(4)
    
    def _get_bin_repr_y_coord(self):
        # 4 bits max (since we have a map with size 8)
        return ((str(bin(self.agent_y_coord)))[2:]).zfill(4)
    
    def _get_bin_repr_gold(self):
        # 1 bit
        return str(bin(self.has_agent_grabbed_gold))[2:]
    
    def _get_bin_repr_climb(self):
        # 1 bit
        return str(bin(self.has_agent_climbed_out))[2:]
    
    def _get_bin_repr_arrow(self):
        # 1 bit
        return str(bin(self.has_agent_shot_arrow))[2:]
        
    def get_index(self):
        """
        Get decimal representation of current state.
        """
        binary_repr = "".join([
            self._get_bin_repr_x_coord(),
            self._get_bin_repr_y_coord(),
            self._get_bin_repr_orientation(),
            self._get_bin_repr_gold(),
            self._get_bin_repr_climb(),
            self._get_bin_repr_arrow()
        ])
        return int(binary_repr, base=2)
    
    @staticmethod
    def get_state_space_size():
        return 2**(13)
