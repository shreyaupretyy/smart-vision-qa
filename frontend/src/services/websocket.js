export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

class WebSocketService {
  constructor() {
    this.ws = null;
    this.callbacks = {};
  }

  connect(videoId, userId) {
    const url = `${WS_URL}/collaborate/${videoId}/${userId}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      if (this.callbacks.onConnect) {
        this.callbacks.onConnect();
      }
    };

    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const handler = this.callbacks[message.type];
      if (handler) {
        handler(message);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (this.callbacks.onError) {
        this.callbacks.onError(error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
      if (this.callbacks.onClose) {
        this.callbacks.onClose();
      }
    };
  }

  on(event, callback) {
    this.callbacks[event] = callback;
  }

  send(type, data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, ...data }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export default new WebSocketService();
