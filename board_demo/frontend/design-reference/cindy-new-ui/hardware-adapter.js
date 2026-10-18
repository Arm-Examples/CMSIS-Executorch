/** Transport-independent hardware boundary. No serial port is opened automatically.
 * Firmware/bridge should translate its actual protocol into these messages.
 */
(() => {
  'use strict';
  const gui = window.ArmEEGDemo;
  if (!gui) throw new Error('Load hardware-adapter.js after the GUI');
  window.ArmEEGHardware = {
    begin() { gui.setSource('hardware'); },
    connected() { gui.setConnection(true); },
    disconnected() { gui.setConnection(false); },
    useMock() { gui.setSource('mock'); },
    receive(input) {
      // Feed one COMPLETE JSON message at a time; transport owns buffering.
      let message;
      try { message = typeof input === 'string' ? JSON.parse(input) : input; }
      catch { return false; }
      if (!message || typeof message !== 'object') return false;
      if (gui.getState().source !== 'hardware') return false;
      switch (message.type) {
        case 'prediction':
          if (message.class_id !== 0 && message.class_id !== 1) return false;
          gui.onBoardPrediction({class_id:message.class_id,ground_truth:message.ground_truth});
          return true;
        case 'ground_truth':
          if (![0,1,null].includes(message.label)) return false;
          gui.onGroundTruth(message.label);
          return true;
        case 'eeg': {
          const channels = message.channels;
          const names=["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"];
          if (!channels || !names.some(n => Array.isArray(channels[n]) && channels[n].length && channels[n].every(Number.isFinite))) return false;
          gui.onEEGFrame(channels);
          return true;
        }
        default: return false;
      }
    }
  };
})();
