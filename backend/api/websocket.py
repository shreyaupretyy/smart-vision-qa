from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        # video_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # websocket -> user_id
        self.user_mapping: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, video_id: str, user_id: str):
        """Connect a client to a video room"""
        await websocket.accept()
        
        if video_id not in self.active_connections:
            self.active_connections[video_id] = set()
        
        self.active_connections[video_id].add(websocket)
        self.user_mapping[websocket] = user_id
        
        logger.info(f"User {user_id} connected to video {video_id}")
        
        # Notify others
        await self.broadcast_to_room(
            video_id,
            {
                "type": "user_joined",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            exclude=websocket
        )
    
    def disconnect(self, websocket: WebSocket, video_id: str):
        """Disconnect a client"""
        if video_id in self.active_connections:
            self.active_connections[video_id].discard(websocket)
            
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]
        
        user_id = self.user_mapping.pop(websocket, None)
        logger.info(f"User {user_id} disconnected from video {video_id}")
        
        return user_id
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        await websocket.send_json(message)
    
    async def broadcast_to_room(
        self, 
        video_id: str, 
        message: dict,
        exclude: WebSocket = None
    ):
        """Broadcast message to all clients in a room"""
        if video_id not in self.active_connections:
            return
        
        disconnected = []
        
        for connection in self.active_connections[video_id]:
            if connection == exclude:
                continue
            
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.active_connections[video_id].discard(connection)
    
    def get_room_users(self, video_id: str) -> list:
        """Get list of users in a room"""
        if video_id not in self.active_connections:
            return []
        
        return [
            self.user_mapping.get(ws, "unknown")
            for ws in self.active_connections[video_id]
        ]


manager = ConnectionManager()


@router.websocket("/ws/collaborate/{video_id}/{user_id}")
async def websocket_collaborate(websocket: WebSocket, video_id: str, user_id: str):
    """
    WebSocket endpoint for real-time collaboration
    
    Supported message types:
    - annotation: Create/update annotation
    - cursor: Update cursor position
    - chat: Send chat message
    - playback: Sync playback position
    """
    await manager.connect(websocket, video_id, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Add metadata
            message["user_id"] = user_id
            message["timestamp"] = datetime.utcnow().isoformat()
            
            message_type = message.get("type")
            
            if message_type == "annotation":
                # Handle annotation creation/update
                await handle_annotation(video_id, message)
                
            elif message_type == "cursor":
                # Broadcast cursor position
                await manager.broadcast_to_room(
                    video_id,
                    {
                        "type": "cursor_update",
                        "user_id": user_id,
                        "x": message.get("x"),
                        "y": message.get("y"),
                        "timestamp": message["timestamp"]
                    },
                    exclude=websocket
                )
                
            elif message_type == "chat":
                # Broadcast chat message
                await manager.broadcast_to_room(
                    video_id,
                    {
                        "type": "chat_message",
                        "user_id": user_id,
                        "message": message.get("message"),
                        "timestamp": message["timestamp"]
                    }
                )
                
            elif message_type == "playback":
                # Sync playback position
                await manager.broadcast_to_room(
                    video_id,
                    {
                        "type": "playback_sync",
                        "user_id": user_id,
                        "timestamp": message.get("video_timestamp"),
                        "playing": message.get("playing", False)
                    },
                    exclude=websocket
                )
            
            elif message_type == "request_users":
                # Send list of users in room
                users = manager.get_room_users(video_id)
                await manager.send_personal_message(
                    {
                        "type": "users_list",
                        "users": users
                    },
                    websocket
                )
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
    
    except WebSocketDisconnect:
        user_id = manager.disconnect(websocket, video_id)
        
        # Notify others
        await manager.broadcast_to_room(
            video_id,
            {
                "type": "user_left",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, video_id)


async def handle_annotation(video_id: str, message: dict):
    """Handle annotation creation/update"""
    # In production, save to database
    annotation = {
        "id": message.get("annotation_id"),
        "video_id": video_id,
        "user_id": message.get("user_id"),
        "frame_number": message.get("frame_number"),
        "timestamp": message.get("video_timestamp"),
        "annotation_type": message.get("annotation_type"),
        "data": message.get("data"),
        "created_at": message.get("timestamp")
    }
    
    logger.info(f"Annotation created: {annotation['id']}")
    
    # Broadcast to all users in room
    await manager.broadcast_to_room(
        video_id,
        {
            "type": "annotation_created",
            "annotation": annotation
        }
    )


@router.websocket("/ws/stream/{video_id}")
async def websocket_stream(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for real-time video streaming
    
    This can be used for processing video frames in real-time
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            
            # Process frame (example: run detection)
            # In production, you'd decode the frame and process it
            
            # Send back results
            result = {
                "type": "frame_processed",
                "timestamp": datetime.utcnow().isoformat(),
                "detections": []  # Placeholder
            }
            
            await websocket.send_json(result)
    
    except WebSocketDisconnect:
        logger.info(f"Stream websocket disconnected for video {video_id}")
    
    except Exception as e:
        logger.error(f"Stream error: {e}")
