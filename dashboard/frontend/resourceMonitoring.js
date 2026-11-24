// Normalize only telemetry returned by the C++ resource simulation.
export function summarizeResourceMonitoring(monitoring = {}) {
  const owners = new Map();
  const events = [];
  for (const event of monitoring.events || []) {
    if (event.type === 'allocated' && event.resource) owners.set(event.resource, event.transactionID);
    if (event.type === 'released' && event.resource) owners.delete(event.resource);
    events.push({ ...event, ownerAfterEvent: event.resource ? (owners.get(event.resource) ?? null) : null });
  }
  return {
    resources: Array.isArray(monitoring.resources) ? monitoring.resources : [],
    events,
    waitingTransactionIDs: [...new Set(monitoring.waitingTransactionIDs || [])],
    owners,
    resourceWaitEvents: Number(monitoring.resourceWaitEvents) || 0,
    lockContentionEvents: Number(monitoring.lockContentionEvents) || 0,
    deadlockDetected: monitoring.deadlockDetected === true,
    deadlockRecovered: monitoring.deadlockRecovered === true,
  };
}
