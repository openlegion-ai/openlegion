/**
 * DashboardWebSocket — auto-reconnecting WebSocket client.
 *
 * Exponential backoff: 1s, 2s, 4s ... up to 30s max.
 * Resets backoff on successful connection.
 */
class DashboardWebSocket {
  constructor(url, { onEvent, onConnect, onDisconnect }) {
    this.url = url;
    this._onEvent = onEvent;
    this._onConnect = onConnect;
    this._onDisconnect = onDisconnect;
    this._ws = null;
    this._backoff = 1000;
    this._maxBackoff = 30000;
    this._intentionalClose = false;
    this._timer = null;
  }

  connect() {
    this._intentionalClose = false;
    try {
      this._ws = new WebSocket(this.url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }

    this._ws.onopen = () => {
      this._backoff = 1000;
      if (this._onConnect) this._onConnect();
    };

    this._ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (this._onEvent) this._onEvent(data);
      } catch (_) {}
    };

    this._ws.onclose = () => {
      if (this._onDisconnect) this._onDisconnect();
      if (!this._intentionalClose) this._scheduleReconnect();
    };

    this._ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  disconnect() {
    this._intentionalClose = true;
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }

  _scheduleReconnect() {
    this._timer = setTimeout(() => {
      this._timer = null;
      this.connect();
    }, this._backoff);
    this._backoff = Math.min(this._backoff * 2, this._maxBackoff);
  }
}
