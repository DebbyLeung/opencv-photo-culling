function cullingApp() {
  return {
    // 預設資料夾路徑
    workflowFolder: '',
    saveSuccess: false,
    autoRunFiltering: true,
    filteringSaveError: '',
    manualFilterBusy: false,
    manualFilterRunning: false,
    manualFilterMessage: '',
    manualFilterError: '',

    photos: [],
    incomingJpgs: [],
    rawFiles: [],
    refreshingMedia: false,
    mediaRefreshError: '',
    jpgExpanded: true,
    filteredExpanded: true,
    rawExpanded: true,
    filter: 'all',
    isConnected: false,
    async init() {
      // 載入目前設定
      const res = await fetch('/api/config');
      const data = await res.json();
      this.workflowFolder = data.workflow_folder;
      this.autoRunFiltering = data.auto_run_filtering;
      this.syncManualFilterBusy();
      await this.refreshMedia();
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
          const score = this.scorePhoto(data);
          const photo = {
            ...data,
            score,
            selected: score > 4
          };
          this.photos.unshift(photo);
          this.refreshMedia();
        } else if (data.event === 'MEDIA_CHANGED') {
          this.refreshMedia();
        }
      };

      socket.onclose = () => {
        this.isConnected = false;
        console.log('[-] WebSocket Disconnected');
      };
    },

    async refreshMedia() {
      if (this.refreshingMedia) return;
      this.refreshingMedia = true;
      this.mediaRefreshError = '';
      try {
        const res = await fetch('/api/media');
        if (!res.ok) throw new Error(`Media refresh failed: ${res.status}`);
        const media = await res.json();
        this.incomingJpgs = media.jpg;
        this.rawFiles = media.raw;
      } catch (err) {
        console.error('Error refreshing incoming media:', err);
        this.mediaRefreshError = 'Could not refresh media.';
      } finally {
        this.refreshingMedia = false;
      }
    },

    async saveWorkflowFolder() {
      if (!this.workflowFolder) return;

      try {
        const params = new URLSearchParams({ workflow_folder: this.workflowFolder });
        const res = await fetch(`/api/config/workflow-folder?${params}`, {
          method: 'POST',
        });

        if (res.ok) {
          const data = await res.json();
          console.log('Workflow folder updated:', data.path);
          this.workflowFolder = data.path;
          this.saveSuccess = true;
          setTimeout(() => this.saveSuccess = false, 3000);
        } else {
          console.error('Failed to update workflow folder');
        }
      } catch (err) {
        console.error('Error updating workflow folder:', err);
      }
    },

    async setAutoRunFiltering() {
      this.syncManualFilterBusy();
      this.filteringSaveError = '';
      try {
        const res = await fetch('/api/config/auto-run-filtering', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: this.autoRunFiltering })
        });
        if (!res.ok) throw new Error(`Filtering setting save failed: ${res.status}`);
        const data = await res.json();
        this.autoRunFiltering = data.auto_run_filtering;
      } catch (err) {
        console.error('Error updating auto-run filtering:', err);
        this.filteringSaveError = 'Could not save filtering setting.';
        try {
          const res = await fetch('/api/config');
          if (!res.ok) throw new Error(`Config reload failed: ${res.status}`);
          const data = await res.json();
          this.autoRunFiltering = data.auto_run_filtering;
        } catch (reloadErr) {
          console.error('Could not reload filtering setting:', reloadErr);
        }
      } finally {
        this.syncManualFilterBusy();
      }
    },

    syncManualFilterBusy() {
      this.manualFilterBusy = this.autoRunFiltering || this.manualFilterRunning;
    },

    async runFilter() {
      if (this.manualFilterBusy) return;
      this.manualFilterRunning = true;
      this.manualFilterBusy = true;
      this.manualFilterMessage = '';
      this.manualFilterError = '';
      try {
        const res = await fetch('/api/filter/run', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Filter run failed: ${res.status}`);
        this.manualFilterMessage = data.processed
          ? `Filtered ${data.processed} incoming JPG(s).`
          : this.autoRunFiltering
            ? 'No pending JPGs; automatic filtering may have already processed them.'
            : 'No incoming JPGs to filter.';
      } catch (err) {
        console.error('Could not start manual filtering:', err);
        this.manualFilterError = err.message || 'Could not start filtering.';
      } finally {
        this.manualFilterRunning = false;
        this.syncManualFilterBusy();
      }
    },

    scorePhoto(photo) {
      const blurScore = Number(photo.blurScore) || 0;
      const earScore = Number(photo.earScore) || 0;
      const blurThreshold = Number(photo.blurThreshold) || 110;
      const earThreshold = Number(photo.earThreshold) || 0.21;
      const blurPassed = blurScore >= blurThreshold;
      const eyesPassed = earScore >= earThreshold;
      if (blurPassed && eyesPassed) return 5;

      const blurQuality = Math.min(blurScore / blurThreshold, 1);
      const eyeQuality = Math.min(earScore / earThreshold, 1);
      return Math.min(4, Math.max(1, Math.ceil(((blurQuality + eyeQuality) / 2) * 4)));
    },

    get scoredPhotos() {
      return [...this.photos].sort((left, right) => right.score - left.score);
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
