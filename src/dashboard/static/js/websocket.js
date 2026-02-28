/**
 * DashboardWebSocket — auto-reconnecting WebSocket client.
 *
 * Exponential backoff: 1s, 2s, 4s ... up to 30s max.
 * Resets backoff on successful connection.
 */
class DashboardWebSocket {
  constructor(url, { onEvent, onConnect, onDisconnect, onReconnectTick }) {
    this.url = url;
    this._onEvent = onEvent;
    this._onConnect = onConnect;
    this._onDisconnect = onDisconnect;
    this._onReconnectTick = onReconnectTick;
    this._ws = null;
    this._backoff = 1000;
    this._maxBackoff = 30000;
    this._intentionalClose = false;
    this._timer = null;
    this._reconnectAt = null; // Epoch ms when reconnect will fire
    this._countdownTimer = null;
    this.reconnectIn = 0; // Seconds until reconnect (exposed for UI)
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
      this._reconnectAt = null;
      this.reconnectIn = 0;
      if (this._countdownTimer) { clearInterval(this._countdownTimer); this._countdownTimer = null; }
      if (this._onReconnectTick) this._onReconnectTick(0);
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
    if (this._countdownTimer) {
      clearInterval(this._countdownTimer);
      this._countdownTimer = null;
    }
    this._reconnectAt = null;
    this.reconnectIn = 0;
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }

  _scheduleReconnect() {
    this._reconnectAt = Date.now() + this._backoff;
    this.reconnectIn = Math.ceil(this._backoff / 1000);
    if (this._onReconnectTick) this._onReconnectTick(this.reconnectIn);
    this._timer = setTimeout(() => {
      this._timer = null;
      this._reconnectAt = null;
      this.reconnectIn = 0;
      if (this._countdownTimer) { clearInterval(this._countdownTimer); this._countdownTimer = null; }
      if (this._onReconnectTick) this._onReconnectTick(0);
      this.connect();
    }, this._backoff);
    // Update countdown every second
    if (this._countdownTimer) clearInterval(this._countdownTimer);
    this._countdownTimer = setInterval(() => {
      if (!this._reconnectAt) { this.reconnectIn = 0; clearInterval(this._countdownTimer); this._countdownTimer = null; if (this._onReconnectTick) this._onReconnectTick(0); return; }
      this.reconnectIn = Math.max(0, Math.ceil((this._reconnectAt - Date.now()) / 1000));
      if (this._onReconnectTick) this._onReconnectTick(this.reconnectIn);
    }, 1000);
    this._backoff = Math.min(this._backoff * 2, this._maxBackoff);
  }
}
