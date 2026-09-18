sim = require 'sim'
local objects
local done = false
local function emit(...)
    local a={...}
    for i=1,#a do a[i]=tostring(a[i]) end
    print('[SBC_AUDIT_V2]\t'..table.concat(a,'\t'))
end
local function safe(f,...)
    local ok,v=pcall(f,...)
    if ok then return tostring(v) end
    return 'ERROR:'..tostring(v)
end
local function param(h,name)
    if sim[name]==nil then return 'API_UNAVAILABLE:'..name end
    return safe(sim.getObjectInt32Param,h,sim[name])
end
local function unique(name)
    local result=nil
    for _,h in ipairs(objects) do
        if sim.getObjectAlias(h)==name then
            if result~=nil then emit('AMBIGUOUS',name);return nil end
            result=h
        end
    end
    if result==nil then emit('MISSING',name) end
    return result
end
local function snapshot(label)
    emit('SNAPSHOT',label,'time',sim.getSimulationTime())
    for _,h in ipairs(objects) do
        local typ=sim.getObjectType(h)
        local name=sim.getObjectAlias(h)
        if typ==sim.object_joint_type then
            emit('JOINT',h,name,'parent',sim.getObjectParent(h),
                'mode',sim.getJointMode(h),'enabled',safe(sim.isDynamicallyEnabled,h),
                'control',param(h,'jointintparam_dynctrlmode'),
                'force_limit',safe(sim.getJointTargetForce,h))
        elseif typ==sim.object_shape_type then
            emit('SHAPE',h,name,'parent',sim.getObjectParent(h),
                'static',param(h,'shapeintparam_static'),
                'respondable',param(h,'shapeintparam_respondable'),
                'enabled',safe(sim.isDynamicallyEnabled,h),
                'mass',safe(sim.getShapeMass,h))
        elseif typ==sim.object_dummy_type then
            emit('DUMMY',h,name,'linked',safe(sim.getLinkDummy,h),
                'link_type',param(h,'dummyintparam_link_type'))
        end
    end
    for i=1,5 do
        local tip=unique('downArm'..i..'Tip')
        local target=unique('downArm'..i..'Target')
        if tip~=nil and target~=nil then
            local p=sim.getObjectPosition(tip,target)
            emit('GEOMETRIC_DISTANCE',i,math.sqrt(p[1]^2+p[2]^2+p[3]^2))
            if sim.getObjectType(tip)==sim.object_dummy_type and
               sim.getObjectType(target)==sim.object_dummy_type then
                emit('LINK',i,sim.getLinkDummy(tip)==target,
                    'tip_type',param(tip,'dummyintparam_link_type'),
                    'target_type',param(target,'dummyintparam_link_type'))
            else
                emit('NOT_DUMMY_PAIR',i,'tip_object_type',sim.getObjectType(tip),
                    'target_object_type',sim.getObjectType(target))
            end
        end
    end
    local ball=sim.getObject('/Sphere',{noError=true})
    if ball~=-1 then
        emit('BALL','type',sim.getObjectType(ball),'enabled',safe(sim.isDynamicallyEnabled,ball),
            'static',param(ball,'shapeintparam_static'),
            'respondable',param(ball,'shapeintparam_respondable'),'mass',safe(sim.getShapeMass,ball))
    else emit('MISSING','/Sphere') end
end
function sysCall_init()
    local root=sim.getObject('/stewartPlatform')
    objects=sim.getObjectsInTree(root,sim.handle_all,0)
    emit('CONSTANTS','dynamic',sim.jointmode_dynamic,'kinematic',sim.jointmode_kinematic)
    snapshot('before_physics')
end
function sysCall_sensing()
    if not done then snapshot('after_first_physics_step');done=true end
end
