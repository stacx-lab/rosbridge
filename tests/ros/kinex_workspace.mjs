// Native cross-repository smoke: start rosbridge, export a workspace, prepare its
// Kinex Python, then: node tests/ros/kinex_workspace.mjs WORKSPACE.
import assert from 'node:assert/strict'
import { access, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { createKinexRuntime } from '../../../kinex2.0/packages/core/dist/index.js'
const directory = path.resolve(process.argv[2])
const runtime = createKinexRuntime({directory, enableRecording:true})
const owned = []
const host = {sessionID:'native-simple',ask:async()=>{}}
const call = (name, args) => runtime.tools.call(name, args, host)
async function exec(body, {error=false, timeout_s=120}={}) {
  const result = await call('exec', {python:`import asyncio
from kinex import communication, finish, helpers, runtime
from kinex.errors import OperationError
async def measured(io, operation, time_s):
    stream = io.subscribe(operation)
    try:
        async with asyncio.timeout(5):
            while True:
                state = await stream.next()
                stamp = state.header.stamp
                if stamp.sec + stamp.nanosec / 1e9 >= time_s - 1e-8:
                    return state
    finally:
        stream.close()
async def clock(io):
    stamp = (await io.subscribe('world.time').next(timeout_s=5)).clock
    return stamp.sec + stamp.nanosec / 1e9
async def main():
    async with communication() as io:
${body.split('\n').map(line=>'        '+line).join('\n')}
asyncio.run(main())`,timeout_s})
  assert.equal(Boolean(result.isError),error,result.output)
  if (error) return result.output
  console.log(result.output)
  return JSON.parse(result.output).result
}
try {
  const profiles = await runtime.devices.list()
  assert.deepEqual(Object.keys(profiles.find(d=>d.name==='world').operations), ['time','wait','stop','reset'])
  assert.deepEqual(Object.keys(profiles.find(d=>d.name==='gripper').operations), ['joint_state','target','stop'])
  for (const name of ['world','arm','gripper','camera_agentview','camera_wrist']) {
    await (await runtime.devices.load(name)).ready
    owned.push(name)
    console.log('loaded',name)
  }
  const started = await runtime.recording.start({specs:{gripper:{rate:20},camera_agentview:{rate:10}},sessionID:host.sessionID})
  assert.deepEqual(started.refusals,{})
  const first = await exec(`assert runtime.describe('gripper.target')['message'] == {'data': 'double'}
assert set(runtime.describe('world.wait')['request']) == {'duration_s'}
assert all(not name.endswith(('/task/status','/task/instruction')) for name,_ in io.node.get_topic_names_and_types())
state = await io.subscribe('arm.joint_state').next(timeout_s=5)
tf = await helpers.lookup_transform(io,'world:base','gripper:tcp',state.header.stamp)
image, depth = await helpers.camera_frame(io,'camera_agentview')
assert depth is None and image.header.frame_id == 'camera_agentview/base'
finish({'time':await clock(io),'stamp':{'sec':state.header.stamp.sec,'nanosec':state.header.stamp.nanosec},'camera':[image.width,image.height],'tcp_z':tf.transform.translation.z})`)
  await exec(`await io.publish('gripper.target',{'data':0.})
finish({'sent':True})`)
  const closed = await exec(`from builtin_interfaces.msg import Time
waited=await io.request('world.wait',duration_s=1.5)
state=await measured(io,'gripper.joint_state',waited.time_s)
# Historical lookup survives the command's separate exec turn.
await helpers.lookup_transform(io,'world:base','gripper:tcp',Time(**${JSON.stringify(first.stamp)}))
finish({'finger':state.position[0]})`)
  assert.ok(closed.finger < .01)
  await exec(`await io.request('gripper.stop')
waited=await io.request('world.wait',duration_s=.3)
state=await measured(io,'gripper.joint_state',waited.time_s)
assert state.position[0] < .01
await io.publish('gripper.target',{'data':.04})
waited=await io.request('world.wait',duration_s=1.5)
state=await measured(io,'gripper.joint_state',waited.time_s)
assert state.position[0] > .03
finish({'open':state.position[0]})`)
  // A failed script stops what it commanded: the gripper holds open instead of closing.
  await exec(`await io.publish('gripper.target',{'data':0.})
raise RuntimeError('intentional failure cleanup')`,{error:true})
  await exec(`waited=await io.request('world.wait',duration_s=.3)
state=await measured(io,'gripper.joint_state',waited.time_s)
assert state.position[0] > .03
finish({'failure_cleanup_held':True})`)
  // Waits are sequential; an abandoned wait stops the world.
  await exec(`pending=asyncio.create_task(io.request('world.wait',duration_s=60))
await asyncio.sleep(.5)
try:
    await io.request('world.wait',duration_s=0)
except OperationError as error:
    assert 'sequential' in error.response.message
else:
    raise AssertionError('concurrent wait was admitted')
pending.cancel()
await asyncio.gather(pending,return_exceptions=True)
before=await clock(io)
await asyncio.sleep(1)
assert await clock(io)==before
finish({'concurrent_wait_rejected':True})`)
  // A script killed mid-wait leaves no wait advancing the world.
  await exec(`await io.request('world.wait',duration_s=60)`,{error:true,timeout_s:3})
  const settled = await exec(`before=await clock(io)
await asyncio.sleep(1)
after=await clock(io)
await io.request('world.wait',duration_s=0)
finish({'advanced_s':after-before})`)
  assert.equal(settled.advanced_s,0)
  // A device stop cancels the running script that commanded it.
  const markers = await mkdtemp(path.join(tmpdir(),'kinex-smoke-'))
  const marker = path.join(markers,'commanded')
  const running = exec(`await io.publish('gripper.target',{'data':.04})
open(${JSON.stringify(marker)},'w').close()
await io.request('world.wait',duration_s=60)`,{error:true})
  while (!await access(marker).then(()=>true,()=>false)) await delay(20)
  assert.ok(!(await call('device_stop',{deviceID:'gripper'})).isError)
  assert.match(await running,/device stop cancelled this invocation/)
  await rm(markers,{recursive:true,force:true})
  // Direct reads reuse the device's loaded connection.
  const latencies = []
  for (let index = 0; index < 5; index++) {
    const started = performance.now()
    assert.ok(!(await call('device',{deviceID:'arm',action:'joint_state'})).isError)
    latencies.push(performance.now() - started)
  }
  console.log('device read ms',latencies.map(value=>value.toFixed(1)).join(' '))
  const recording = await runtime.recording.stop()
  console.log('recording',JSON.stringify(recording))
  await runtime.devices.unload('gripper')
  await exec(`try:
    await helpers.lookup_transform(io,'world:base','gripper:tcp',timeout_s=.1)
except Exception:
    finish({'unloaded_frame_unavailable':True})
else:
    raise AssertionError('unloaded frame remained available')`)
  await (await runtime.devices.load('gripper')).ready
  await exec(`tf=await helpers.lookup_transform(io,'world:base','gripper:tcp')
finish({'reloaded_frame':tf.child_frame_id})`)
} finally {
  for (const name of owned.reverse()) {
    try {await runtime.devices.unload(name)} catch(error) {console.error('unload',name,error);process.exitCode=1}
  }
  await runtime.dispose()
}
