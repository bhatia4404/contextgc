// Red "watch the pitch" button with a pulsing play icon; links to the YouTube video.
export const PITCH_URL = "https://www.youtube.com/watch?v=gwRTK6PQHUY";

export default function WatchPitch({ big = false }: { big?: boolean }) {
  return (
    <a className={`yt-btn${big ? " big" : ""}`} href={PITCH_URL} target="_blank" rel="noopener noreferrer"
       aria-label="Watch the 3-minute ContextGC pitch on YouTube">
      <span className="yt-play" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="12" height="12"><path d="M8 5v14l11-7z" fill="currentColor" /></svg>
      </span>
      {big ? "Watch the 3-minute pitch" : "Watch the pitch"}
    </a>
  );
}
