import { useEffect, useState } from 'react';

// Runs fn() whenever deps change. Set enabled=false to skip the request.
export default function useFetch(fn, deps = [], enabled = true) {
  const [state, setState] = useState({ data: null, loading: enabled, error: null });
  const [tries, setTries] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, loading: false, error: null });
      return undefined;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn()
      .then((data) => !cancelled && setState({ data, loading: false, error: null }))
      .catch((error) => !cancelled && setState({ data: null, loading: false, error }));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, tries]);

  return { ...state, refetch: () => setTries((t) => t + 1) };
}
