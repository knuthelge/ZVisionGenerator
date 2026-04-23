import type { SSEEvent, BatchCompletedEvent } from '$lib/types';

export type SSEEventHandler<T extends SSEEvent = SSEEvent> = (event: T) => void;

export interface SSESubscription {
  close: () => void;
}

export function connectJobSSE(
  jobId: string,
  handlers: {
    onStep?: SSEEventHandler;
    onBatchCompleted?: SSEEventHandler<BatchCompletedEvent>;
    onJobCompleted?: SSEEventHandler;
    onJobFailed?: SSEEventHandler;
    onJobCancelled?: SSEEventHandler;
    onProgressText?: SSEEventHandler;
    onJobPaused?: SSEEventHandler;
    onJobResumed?: SSEEventHandler;
    onStatus?: (type: string, data: SSEEvent) => void;
    onClose?: () => void;
  }
): SSESubscription {
  const es = new EventSource(`/jobs/${jobId}/events`);
  const TERMINAL_EVENTS: ReadonlySet<string> = new Set(['job_completed', 'job_failed', 'batch_cancelled']);

  function handleEvent(type: string, data: SSEEvent): void {
    switch (type) {
      case 'step_progress': handlers.onStep?.(data); break;
      case 'model_loading':
      case 'batch_started':
      case 'workflow_stage_started':
      case 'workflow_stage_completed':
      case 'generation_finished':
        handlers.onStatus?.(type, data);
        break;
      case 'batch_completed': {
        const ev = data as BatchCompletedEvent;
        handlers.onBatchCompleted?.(ev);
        // Only close if this is the LAST batch in a multi-run job
        // (job_completed is the reliable terminal event — this is informational)
        break;
      }
      case 'job_completed': handlers.onJobCompleted?.(data); break;
      case 'job_failed': handlers.onJobFailed?.(data); break;
      case 'batch_cancelled': handlers.onJobCancelled?.(data); break;
      case 'progress_text': handlers.onProgressText?.(data); break;
      case 'job_paused': handlers.onJobPaused?.(data); break;
      case 'job_resumed': handlers.onJobResumed?.(data); break;
    }

    if (TERMINAL_EVENTS.has(type)) {
      es.close();
      handlers.onClose?.();
    }
  }

  const eventTypes = ['step_progress', 'batch_completed', 'job_completed', 'job_failed', 'batch_cancelled', 'progress_text', 'job_paused', 'job_resumed', 'model_loading', 'batch_started', 'workflow_stage_started', 'workflow_stage_completed', 'generation_finished'];
  eventTypes.forEach(type => {
    es.addEventListener(type, (event: Event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as SSEEvent;
        handleEvent(type, data);
      } catch {
        // ignore malformed events
      }
    });
  });

  es.onerror = () => {
    es.close();
    handlers.onClose?.();
  };

  return { close: () => { es.close(); handlers.onClose?.(); } };
}
