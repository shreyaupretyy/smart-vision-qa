export default function VideoPlayer({ videoId, metadata }) {
  const videoUrl = `http://localhost:8000/uploads/${videoId}.mp4`;

  return (
    <div className="card">
      <div className="aspect-video bg-black rounded-lg overflow-hidden">
        <video
          controls
          className="w-full h-full"
          src={videoUrl}
        >
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
}
