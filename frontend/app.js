function cullingApp() {
  return {
    // 預設資料夾路徑
    watchedFolder: '',
    destinationFolder: '',
    saveSuccess: false,

    photos: [],
    filter: 'all',
    isConnected: false,
    async init() {
      // 載入目前設定
      const res = await fetch('/api/config');
      const data = await res.json();
      this.watchedFolder = data.watched_folder;
      this.destinationFolder = data.destination_folder;
      // Connects to ws://localhost:8000/ws dynamically based on current page host
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws`;

      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        this.isConnected = true;
        console.log('[+] Connected to Culling WebSocket Server');
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.event === 'NEW_PHOTO') {
          // Prepend newly culled photo to the dashboard grid
          this.photos.unshift(data);
        }
      };

      socket.onclose = () => {
        this.isConnected = false;
        console.log('[-] WebSocket Disconnected');
      };
    },
    async saveConfig() {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          watched_folder: this.watchedFolder,
          destination_folder: this.destinationFolder
        })
      });
      if (res.ok) {
        this.saveSuccess = true;
        setTimeout(() => this.saveSuccess = false, 3000);
      }
    },

    get filteredPhotos() {
      if (this.filter === 'passed') return this.photos.filter(p => p.status === 'passed');
      if (this.filter === 'rejected') return this.photos.filter(p => p.status === 'rejected');
      return this.photos;
    },

    get passedCount() {
      return this.photos.filter(p => p.status === 'passed').length;
    },

    get rejectedCount() {
      return this.photos.filter(p => p.status === 'rejected').length;
    },

    togglePhotoStatus(photo) {
      photo.status = photo.status === 'passed' ? 'rejected' : 'passed';
    }
  };
}
