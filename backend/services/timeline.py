from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TimelineGenerator:
    """Event timeline generation service with AI summarization and clustering"""
    
    def __init__(self):
        """Initialize timeline generator"""
        pass
    
    def generate_timeline(
        self,
        video_id: str,
        detections: List[Dict],
        captions: List[Dict],
        transcripts: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate comprehensive event timeline
        
        Args:
            video_id: Video ID
            detections: Object detection results
            captions: Frame captions
            transcripts: Audio transcription segments (optional)
            
        Returns:
            Timeline dictionary with events and summary
        """
        events = []
        
        # Extract events from detections
        detection_events = self._extract_detection_events(detections)
        events.extend(detection_events)
        
        # Extract events from captions
        caption_events = self._extract_caption_events(captions)
        events.extend(caption_events)
        
        # Extract events from transcripts
        if transcripts:
            transcript_events = self._extract_transcript_events(transcripts)
            events.extend(transcript_events)
        
        # Sort events by timestamp
        events.sort(key=lambda x: x["timestamp"])
        
        # Cluster similar events
        clustered_events = self._cluster_events(events)
        
        # Generate summary
        summary = self._generate_summary(clustered_events)
        
        return {
            "video_id": video_id,
            "total_events": len(events),
            "events": clustered_events,
            "summary": summary,
            "timeline_generated": datetime.utcnow().isoformat()
        }
    
    def _extract_detection_events(self, detections: List[Dict]) -> List[Dict]:
        """Extract events from object detections"""
        events = []
        
        # Group detections by frame
        frames = {}
        for det in detections:
            frame_num = det.get("frame_number", 0)
            if frame_num not in frames:
                frames[frame_num] = []
            frames[frame_num].append(det)
        
        # Create events for significant detection changes
        prev_objects = set()
        
        for frame_num in sorted(frames.keys()):
            frame_dets = frames[frame_num]
            current_objects = set(d["class_name"] for d in frame_dets)
            
            # New objects appeared
            new_objects = current_objects - prev_objects
            if new_objects:
                timestamp = frame_dets[0].get("timestamp", 0.0)
                events.append({
                    "timestamp": timestamp,
                    "frame_number": frame_num,
                    "event_type": "object_appeared",
                    "description": f"New objects detected: {', '.join(new_objects)}",
                    "confidence": 0.8,
                    "data": {"objects": list(new_objects)}
                })
            
            # Objects disappeared
            disappeared = prev_objects - current_objects
            if disappeared:
                timestamp = frame_dets[0].get("timestamp", 0.0)
                events.append({
                    "timestamp": timestamp,
                    "frame_number": frame_num,
                    "event_type": "object_disappeared",
                    "description": f"Objects no longer visible: {', '.join(disappeared)}",
                    "confidence": 0.8,
                    "data": {"objects": list(disappeared)}
                })
            
            prev_objects = current_objects
        
        logger.info(f"Extracted {len(events)} detection events")
        return events
    
    def _extract_caption_events(self, captions: List[Dict]) -> List[Dict]:
        """Extract events from frame captions"""
        events = []
        
        # Look for significant caption changes
        prev_caption = ""
        
        for cap in captions:
            caption_text = cap.get("caption", "")
            frame_num = cap.get("frame_number", 0)
            timestamp = cap.get("timestamp", 0.0)
            
            # Check if caption is significantly different
            if caption_text and caption_text != prev_caption:
                # Simple change detection (can be enhanced with NLP)
                similarity = self._calculate_text_similarity(prev_caption, caption_text)
                
                if similarity < 0.7:  # Significant change
                    events.append({
                        "timestamp": timestamp,
                        "frame_number": frame_num,
                        "event_type": "scene_change",
                        "description": caption_text,
                        "confidence": 0.75,
                        "data": {"caption": caption_text}
                    })
            
            prev_caption = caption_text
        
        logger.info(f"Extracted {len(events)} caption events")
        return events
    
    def _extract_transcript_events(self, transcripts: List[Dict]) -> List[Dict]:
        """Extract events from transcripts"""
        events = []
        
        for segment in transcripts:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            timestamp = segment.get("start", 0.0)
            
            # Detect question marks (questions)
            if "?" in text:
                events.append({
                    "timestamp": timestamp,
                    "frame_number": int(timestamp * 30),  # Approximate
                    "event_type": "question",
                    "description": f"Question asked: {text}",
                    "confidence": 0.9,
                    "data": {"text": text}
                })
            
            # Detect keywords for important events
            keywords = ["important", "attention", "notice", "alert", "warning"]
            if any(kw in text.lower() for kw in keywords):
                events.append({
                    "timestamp": timestamp,
                    "frame_number": int(timestamp * 30),
                    "event_type": "important_speech",
                    "description": f"Important: {text}",
                    "confidence": 0.85,
                    "data": {"text": text}
                })
        
        logger.info(f"Extracted {len(events)} transcript events")
        return events
    
    def _cluster_events(
        self, 
        events: List[Dict],
        time_threshold: float = 2.0
    ) -> List[Dict]:
        """
        Cluster events that occur close in time
        
        Args:
            events: List of events
            time_threshold: Time window for clustering (seconds)
            
        Returns:
            Clustered events
        """
        if not events:
            return []
        
        clustered = []
        current_cluster = [events[0]]
        
        for event in events[1:]:
            # Check if event is close to cluster
            if event["timestamp"] - current_cluster[0]["timestamp"] <= time_threshold:
                current_cluster.append(event)
            else:
                # Merge current cluster and start new one
                merged = self._merge_event_cluster(current_cluster)
                clustered.append(merged)
                current_cluster = [event]
        
        # Add last cluster
        if current_cluster:
            merged = self._merge_event_cluster(current_cluster)
            clustered.append(merged)
        
        logger.info(f"Clustered {len(events)} events into {len(clustered)} groups")
        return clustered
    
    def _merge_event_cluster(self, cluster: List[Dict]) -> Dict:
        """Merge multiple events into a single event"""
        if len(cluster) == 1:
            return cluster[0]
        
        # Combine descriptions
        descriptions = [e["description"] for e in cluster]
        combined_desc = "; ".join(descriptions)
        
        # Average confidence
        confidences = [e["confidence"] for e in cluster]
        avg_confidence = sum(confidences) / len(confidences)
        
        return {
            "timestamp": cluster[0]["timestamp"],
            "frame_number": cluster[0]["frame_number"],
            "event_type": "combined",
            "description": combined_desc,
            "confidence": avg_confidence,
            "data": {
                "sub_events": cluster,
                "event_count": len(cluster)
            }
        }
    
    def _generate_summary(self, events: List[Dict]) -> str:
        """Generate text summary of timeline"""
        if not events:
            return "No significant events detected."
        
        # Count event types
        event_types = Counter(e["event_type"] for e in events)
        
        # Build summary
        summary_parts = [
            f"Video contains {len(events)} significant events.",
        ]
        
        # Most common events
        most_common = event_types.most_common(3)
        if most_common:
            event_desc = ", ".join(f"{count} {etype}" for etype, count in most_common)
            summary_parts.append(f"Most frequent events: {event_desc}.")
        
        # Time span
        if len(events) > 1:
            duration = events[-1]["timestamp"] - events[0]["timestamp"]
            summary_parts.append(
                f"Events span {duration:.1f} seconds from "
                f"{self._format_time(events[0]['timestamp'])} to "
                f"{self._format_time(events[-1]['timestamp'])}."
            )
        
        return " ".join(summary_parts)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0-1)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def filter_events_by_type(
        self, 
        events: List[Dict], 
        event_types: List[str]
    ) -> List[Dict]:
        """Filter events by type"""
        return [e for e in events if e["event_type"] in event_types]
    
    def get_events_in_range(
        self,
        events: List[Dict],
        start_time: float,
        end_time: float
    ) -> List[Dict]:
        """Get events within time range"""
        return [
            e for e in events 
            if start_time <= e["timestamp"] <= end_time
        ]
    
    def find_peak_activity_periods(
        self,
        events: List[Dict],
        window_size: float = 10.0,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find time periods with most activity
        
        Args:
            events: List of events
            window_size: Time window in seconds
            top_k: Number of peak periods to return
            
        Returns:
            List of peak activity periods
        """
        if not events:
            return []
        
        # Create sliding windows
        max_time = max(e["timestamp"] for e in events)
        windows = []
        
        current_time = 0.0
        while current_time <= max_time:
            window_events = self.get_events_in_range(
                events, 
                current_time, 
                current_time + window_size
            )
            
            if window_events:
                windows.append({
                    "start_time": current_time,
                    "end_time": current_time + window_size,
                    "event_count": len(window_events),
                    "events": window_events
                })
            
            current_time += window_size / 2  # 50% overlap
        
        # Sort by event count
        windows.sort(key=lambda x: x["event_count"], reverse=True)
        
        return windows[:top_k]
