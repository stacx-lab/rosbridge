// Validate exported workspaces with Kinex's own profile schema and readiness plan:
//   node tests/kinex_profiles.mjs KINEX_CORE_DIST WORKSPACE...
// Prints one {robot, devices, items} JSON object per workspace; exits nonzero when a
// profile fails to parse.
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [dist, ...workspaces] = process.argv.slice(2).map(value => path.resolve(value))
const load = (file) => import(pathToFileURL(path.join(dist, file)).href)
const { ProfileStore } = await load('device/profile/store.js')
const { readinessPlan } = await load('device/profile/readiness.js')
const results = []
for (const directory of workspaces) {
  const snapshot = await ProfileStore.snapshot(directory)
  const plan = await readinessPlan(directory, snapshot)
  results.push({
    robot: snapshot.robot?.identity?.id ?? null,
    devices: Object.fromEntries(snapshot.devices.map(d => [d.name, {type: d.type, operations: Object.keys(d.operations)}])),
    items: plan.items.map(({id, status, detail}) => ({id, status, detail})),
  })
}
console.log(JSON.stringify(results, null, 2))
