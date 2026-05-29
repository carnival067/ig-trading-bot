export function LoadingSpinner({ message = 'Loading...' }: { message?: string }) {
  return (
    <div className="loading-spinner" role="status" aria-label={message}>
      <div className="spinner" />
      <p>{message}</p>
    </div>
  );
}
