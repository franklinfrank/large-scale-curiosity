local maze_generation = require 'dmlab.system.maze_generation'
local tensor = require 'dmlab.system.tensor'
local log = require 'common.log'
local random = require 'common.random'
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local custom_observations = require 'decorators.custom_observations'
local setting_overrides = require 'decorators.setting_overrides'
local os = require 'os'

local api = {}
local mazes = {}

function api:init(settings)

    -- Read the mazes into the maze variable
    local rows, cols = 21, 21
    
    local seed = os.time()
    random:seed(seed)
    local i = random:uniformInt(1, 99)
    local maze = maze_generation.mazeGeneration{height = rows, width = cols}

    -- Load the file data
    local ents_file = io.open("demos/map_generation/dense_mazes/maze_ents_" .. tostring(i) .. ".txt", "r")
    local ents = ents_file:read("*a")
    local vars_file = io.open("demos/map_generation/dense_mazes/maze_vars_" .. tostring(i) .. ".txt", "r")
    local vars = vars_file:read("*a")

    print("Loading map:", "maze_ents_" .. tostring(i) .. ".txt")
    print(ents)
    print(vars)

    -- Load the entity and variation layers from files
    for j=1,rows do
        for k=1,cols do
            maze:setEntityCell(j, k, ents:sub((j-1)*cols+k,(j-1)*cols+k))
            maze:setVariationsCell(j,k, vars:sub((j-1)*cols+k,(j-1)*cols+k))
        end
    end

    api._maze_name = make_map.makeMap{
        mapName = "gen_21x21_" .. tostring(i),
        mapEntityLayer = maze:entityLayer(),
        mapVariationsLayer = maze:variationsLayer(),
        decalFrequency = -1,
    }

end

local current_maze = 0

function api:start(episode, seed)
    local rows, cols = 21, 21
    random:seed(seed)
    local i = random:uniformInt(1, 99)
    local maze = maze_generation.mazeGeneration{height = rows, width = cols}

    -- Load the file data
    local ents_file = io.open("demos/map_generation/dense_mazes/maze_ents_" .. tostring(i) .. ".txt", "r")
    local ents = ents_file:read("*a")
    local vars_file = io.open("demos/map_generation/dense_mazes/maze_vars_" .. tostring(i) .. ".txt", "r")
    local vars = vars_file:read("*a")

    print("Loading map:", "maze_ents_" .. tostring(i) .. ".txt")
    print(ents)
    print(vars)

    -- Load the entity and variation layers from files
    for j=1,rows do
        for k=1,cols do
            maze:setEntityCell(j, k, ents:sub((j-1)*cols+k,(j-1)*cols+k))
            maze:setVariationsCell(j,k, vars:sub((j-1)*cols+k,(j-1)*cols+k))
        end
    end

    api._maze_name = make_map.makeMap{
        mapName = "gen_21x21_" .. tostring(i),
        mapEntityLayer = maze:entityLayer(),
        mapVariationsLayer = maze:variationsLayer(),
        decalFrequency = -1,
    }


    api._current_maze_name = api._maze_name
end


function api:nextMap()
    local mapName = api._current_maze_name
    api._current_maze_name = api._maze_name
    return mapName
end

function api:createPickup(classname)
  return pickups.defaults[classname]
end

function api:updateSpawnVars(spawnVars)
  if spawnVars.classname == 'info_player_start' then
    spawnVars.angle = '90'
    spawnVars.randomAngleRange = '0'
  end
  return spawnVars
end

custom_observations.decorate(api)
setting_overrides.decorate{
    api = api,
    apiParams = {episodeLengthSeconds = 180, camera = {750, 750, 750}},
    decorateWithTimeout = true
}

return api
