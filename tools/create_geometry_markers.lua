-- One-shot commissioning helper for the ORIGINAL kinematic Stewart scene.
-- Run only on a copy while the simulation is stopped, save the scene, then
-- remove/disable this script. It creates measurement markers; it does not
-- convert the mechanism to dynamics.

sim = require 'sim'

local function allObjects()
    return sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
end

local function uniqueAlias(name)
    local found = nil
    for _, h in ipairs(allObjects()) do
        if sim.getObjectAlias(h) == name then
            if found ~= nil then
                error('Alias is not unique: ' .. name)
            end
            found = h
        end
    end
    if found == nil then error('Missing alias: ' .. name) end
    return found
end

local function existingAlias(name)
    for _, h in ipairs(allObjects()) do
        if sim.getObjectAlias(h) == name then return h end
    end
    return -1
end

local function createMarker(name, source, parent)
    if existingAlias(name) ~= -1 then
        error('Refusing to overwrite existing marker: ' .. name)
    end
    local pose = sim.getObjectPose(source, sim.handle_world)
    local marker = sim.createDummy(0.008)
    sim.setObjectAlias(marker, name)
    sim.setObjectPose(marker, sim.handle_world, pose)
    sim.setObjectParent(marker, parent, true)
    return marker
end

local function childWithAlias(parent, name)
    -- bit0: exclude parent; bit1: direct children only
    for _, h in ipairs(sim.getObjectsInTree(parent, sim.handle_all, 3)) do
        if sim.getObjectAlias(h) == name then return h end
    end
    error('Missing child alias ' .. name .. ' under ' .. sim.getObjectAlias(parent))
end

if sim.getSimulationState() ~= sim.simulation_stopped then
    error('Run this helper from a stopped-scene sandbox script, not during simulation')
end

local root = sim.getObject('/stewartPlatform')
local plate = uniqueAlias('platformTable')
local plateFrame = uniqueAlias('tip')

-- Public actuator order is [motor2,motor3,motor4,motor5,motor6,motor1].
for i = 1, 5 do
    local baseSource = uniqueAlias('downArm' .. i .. 'Target')
    local platformSource = uniqueAlias('downArm' .. i .. 'Sphere')
    createMarker('baseAnchor' .. (i + 1), baseSource, root)
    createMarker('platformAnchor' .. (i + 1), platformSource, plateFrame)
end

-- The lower master sphere is a direct child of the model root. The upper
-- master sphere is the direct parent of platformTable in this scene.
local baseMaster = childWithAlias(root, 'upArmSphere')
local platformMaster = sim.getObjectParent(plate)
if sim.getObjectAlias(platformMaster) ~= 'upArmSphere' then
    error('Unexpected master-platform attachment object')
end
createMarker('baseAnchor1', baseMaster, root)
createMarker('platformAnchor1', platformMaster, plateFrame)

print('[SBC_MARKERS] Created 12 geometry markers successfully.')
print('[SBC_MARKERS] Save as stewart_platform_ideal.ttt.')
