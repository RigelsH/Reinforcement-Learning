def rotate_clockwise(orientation):
    """
    [0,1] (north) -> [1,0] (east)
    [1,0] (east) -> [0,-1] (south)
    [0,-1] (south) -> [-1,0] (west)
    [-1,0] (west) -> [0,1] (north)
    """
    return (orientation[1], -orientation[0])

def rotate_counterclockwise(orientation):
    """
    [0,1] (north) -> [-1,0] (west)
    [-1,0] (west) -> [0,-1] (south)
    [0,-1] (south) -> [1,0] (east)
    [1,0] (east) -> [0,1] (north)
    """
    return (-orientation[1], orientation[0])


id2actions = {
    0: 'MOVE',
    1: 'RIGHT',
    2: 'LEFT',
    3: 'SHOOT',
    4: 'GRAB',
    5: 'CLIMB'
}