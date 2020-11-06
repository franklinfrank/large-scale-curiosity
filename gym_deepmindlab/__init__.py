from gym.envs.registration import register 
from gym import make

LEVELS = ['lt_chasm', 'lt_hallway_slope', 'lt_horseshoe_color', 'lt_space_bounce_hard', \
'nav_maze_random_goal_01','nav_maze_random_goal_02', 'nav_maze_random_goal_03', 'nav_maze_random_goal_01_no_apples', 'nav_maze_random_goal_02_no_apples', 'nav_maze_static_01', \
'nav_maze_static_02', 'seekavoid_arena_01', 'stairway_to_melon', 'generated_maze_dense', 'generated_maze_sparse', 'generated_maze_no', 'generated_maze_sparse_rand']

TEST_SUFFIXES = ['NEW', 'NEW4', 'NEWA', 'NEW4A', 'NEWno', 'NEW4no', 'NEWb', 'NEW4b', 'NEWA_DENSER', 'NEW4A_DENSER', 'NEWno_densest']

for rew_type in ['dense', 'sparse', 'no']:
    for i in range(1, 11):
        LEVELS.append("{}_reward_gen_maze_{}".format(rew_type, i))
LEVELS.append("dense_reward_gen_maze_155")
LEVELS.append("dense_reward_gen_maze_156")
for i in TEST_SUFFIXES:
    LEVELS.append("naren_manual_eliza_{}".format(i))
for i in range(1,6):
    LEVELS.append("naren_manual_eliza_NEW4_var{}".format(i))
    LEVELS.append("naren_manual_eliza_NEW4A_var{}".format(i))
    LEVELS.append("naren_manual_eliza_NEW_var{}".format(i))
    LEVELS.append("naren_manual_eliza_NEWA_var{}".format(i))

for rew_type in ['dense', 'sparse', 'no']:
    for i in range(20):
        LEVELS.append("validation_maze_{}_{}".format(rew_type, i))
    
def _to_pascal(text):
    return ''.join(map(lambda x: x.capitalize(), text.split('_')))
    
MAP = { _to_pascal(l):l for l in LEVELS }

for key, l in MAP.items():
    register(
        id='DeepmindLab%s-v0' % key ,
        entry_point='gym_deepmindlab.env:DeepmindLabEnv',
        kwargs = dict(scene = l)
    )
    print('DeepmindLab%s-v0' % key)
    register(
        id='DeepmindLab%sWithDepth-v0' % key,
        entry_point='gym_deepmindlab.env:DeepmindLabEnv',
        kwargs = dict(scene = l, colors = 'RGBD_INTERLEAVED')
    )
    print('DeepmindLab%sWithDepth-v0' % key)
