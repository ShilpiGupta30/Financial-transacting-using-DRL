import test from 'node:test';
import assert from 'node:assert/strict';
import { summarizeResourceMonitoring } from './resourceMonitoring.js';

test('resource monitor replays ownership transitions and deduplicates waiting transactions', () => {
  const view = summarizeResourceMonitoring({
    resources: [{ resourceID: 1, name: 'SHARED', totalUnits: 1, availableUnits: 1 }],
    events: [
      { type: 'allocated', resource: 'SHARED', transactionID: 907 },
      { type: 'resource_wait', resource: 'SHARED', transactionID: 241 },
      { type: 'released', resource: 'SHARED', transactionID: 907 },
    ],
    waitingTransactionIDs: [241, 241], resourceWaitEvents: 2,
    lockContentionEvents: 1, deadlockDetected: true, deadlockRecovered: true,
  });
  assert.equal(view.resources[0].availableUnits, 1);
  assert.equal(view.owners.has('SHARED'), false);
  assert.deepEqual(view.waitingTransactionIDs, [241]);
  assert.equal(view.resourceWaitEvents, 2);
  assert.equal(view.lockContentionEvents, 1);
  assert.equal(view.deadlockDetected, true);
});

test('resource monitor renders empty telemetry safely', () => {
  const view = summarizeResourceMonitoring();
  assert.deepEqual(view.resources, []);
  assert.deepEqual(view.events, []);
  assert.deepEqual(view.waitingTransactionIDs, []);
  assert.equal(view.deadlockDetected, false);
});
