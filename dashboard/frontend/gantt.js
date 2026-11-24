export function buildGanttData(segments = [], requestedMakespan = 0) {
  if (!Array.isArray(segments)) return { segments: [], makespan: 0, error: 'Invalid timeline data.' };
  if (segments.some((segment) => !segment || typeof segment !== 'object')) {
    return { segments: [], makespan: 0, error: 'Timeline contains an invalid segment.' };
  }
  const ordered = [...segments].sort((left, right) =>
    left.startTime - right.startTime || left.order - right.order,
  );
  if (!ordered.length) {
    return { segments: [], makespan: Math.max(0, Number(requestedMakespan) || 0), error: null };
  }
  const result = [];
  let cursor = 0;
  for (const segment of ordered) {
    const startTime = Number(segment.startTime);
    const endTime = Number(segment.endTime);
    if (!Number.isFinite(startTime) || !Number.isFinite(endTime) || startTime < 0 || endTime <= startTime) {
      return { segments: [], makespan: 0, error: 'Timeline contains invalid time intervals.' };
    }
    if (startTime < cursor) return { segments: [], makespan: 0, error: 'Timeline intervals overlap.' };
    if (startTime > cursor) {
      result.push({ order: result.length + 1, transactionID: null, processID: null,
        startTime: cursor, endTime: startTime, duration: startTime - cursor, isIdle: true });
    }
    result.push({ ...segment, order: result.length + 1, startTime, endTime,
      duration: endTime - startTime, isIdle: Boolean(segment.isIdle) });
    cursor = endTime;
  }
  const makespan = Math.max(cursor, Number(requestedMakespan) || 0);
  if (makespan > cursor) {
    result.push({ order: result.length + 1, transactionID: null, processID: null,
      startTime: cursor, endTime: makespan, duration: makespan - cursor, isIdle: true });
  }
  return { segments: result, makespan, error: null };
}
