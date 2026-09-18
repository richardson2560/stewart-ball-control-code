-- Set false for stewart_platform_ideal.ttt and true for the rebuilt dynamics scene.
local EXPECT_DYNAMIC = false

sim = require 'sim'
local checked = false
local failures = {}

local function fail(message)
    failures[#failures + 1] = message
    print('[SBC_GATE][FAIL] ' .. message)
end

local function objects()
    return sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
end

local function unique(name)
    local matches = {}
    for _, h in ipairs(objects()) do
        if sim.getObjectAlias(h) == name then matches[#matches + 1] = h end
    end
    if #matches ~= 1 then
        fail(name .. ': expected one object, found ' .. #matches)
        return -1
    end
    return matches[1]
end

local function intParam(h, p)
    local ok, value = pcall(sim.getObjectInt32Param, h, p)
    if not ok then
        fail(sim.getObjectAlias(h) .. ': parameter read failed: ' .. tostring(value))
        return nil
    end
    return value
end

local function requireShapeState(name, requireMovable)
    local h = unique(name)
    if h == -1 then return end
    if sim.getObjectType(h) ~= sim.object_shape_type then
        fail(name .. ': not a shape')
        return
    end
    local static = intParam(h, sim.shapeintparam_static)
    local respondable = intParam(h, sim.shapeintparam_respondable)
    if requireMovable and static ~= 0 then fail(name .. ': still static') end
    if respondable ~= 1 then fail(name .. ': not respondable') end
    if not sim.isDynamicallyEnabled(h) then fail(name .. ': not dynamically enabled') end
end

local function checkMotor(i)
    local h = unique('motor' .. i)
    if h == -1 then return end
    local expected = EXPECT_DYNAMIC and sim.jointmode_dynamic or sim.jointmode_kinematic
    if sim.getJointMode(h) ~= expected then
        fail('motor' .. i .. ': wrong joint mode')
    end
    if EXPECT_DYNAMIC then
        if not sim.isDynamicallyEnabled(h) then fail('motor' .. i .. ': not dynamically enabled') end
        local ctrl = intParam(h, sim.jointintparam_dynctrlmode)
        if ctrl ~= sim.jointdynctrl_position then fail('motor' .. i .. ': not in position control') end
        local force = sim.getJointTargetForce(h)
        if force == nil or math.abs(force) <= 0 then fail('motor' .. i .. ': invalid force limit') end
    end
end

local function checkClosure(i)
    local tip = unique('downArm' .. i .. 'Tip')
    local target = unique('downArm' .. i .. 'Target')
    if tip == -1 or target == -1 then return end
    local p = sim.getObjectPosition(tip, target)
    local d = math.sqrt(p[1]^2 + p[2]^2 + p[3]^2)
    print(string.format('[SBC_GATE] closure_%d_distance_m=%.9g', i, d))
    if d > 5e-4 then fail('closure ' .. i .. ': exceeds 0.5 mm') end
    if EXPECT_DYNAMIC then
        if sim.getObjectType(tip) ~= sim.object_dummy_type or
           sim.getObjectType(target) ~= sim.object_dummy_type then
            fail('closure ' .. i .. ': dynamics requires a dummy pair')
            return
        end
        if sim.getLinkDummy(tip) ~= target or sim.getLinkDummy(target) ~= tip then
            fail('closure ' .. i .. ': dummy link is not reciprocal')
        end
        local a = intParam(tip, sim.dummyintparam_link_type)
        local b = intParam(target, sim.dummyintparam_link_type)
        if a ~= sim.dummy_linktype_dynamics_loop_closure or
           b ~= sim.dummy_linktype_dynamics_loop_closure then
            fail('closure ' .. i .. ': wrong dummy link type')
        end
    end
end

function sysCall_init()
    print('[SBC_GATE] expected_mode=' .. (EXPECT_DYNAMIC and 'dynamic' or 'ideal_kinematic'))
end

function sysCall_sensing()
    if checked then return end
    checked = true

    for i = 1, 6 do checkMotor(i) end
    requireShapeState('Sphere', true)
    requireShapeState('platformTable', EXPECT_DYNAMIC)
    for i = 1, 5 do checkClosure(i) end

    if EXPECT_DYNAMIC then
        local root = unique('stewartPlatform')
        if root ~= -1 then
            for _, h in ipairs(sim.getObjectsInTree(root, sim.object_joint_type, 0)) do
                if sim.getJointMode(h) ~= sim.jointmode_dynamic then
                    fail(sim.getObjectAlias(h) .. ': a mechanism joint is not dynamic')
                elseif not sim.isDynamicallyEnabled(h) then
                    fail(sim.getObjectAlias(h) .. ': dynamic joint is not enabled')
                end
            end
        end
    end

    if #failures == 0 then
        print('[SBC_GATE][PASS] scene accepted for the selected validation route')
    else
        print('[SBC_GATE][SUMMARY] failures=' .. #failures)
    end
    sim.stopSimulation()
end

