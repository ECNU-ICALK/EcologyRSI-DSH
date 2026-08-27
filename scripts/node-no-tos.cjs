// macOS can reject the optional IP_TOS/DSCP socket setting used by Node's
// undici client with EINVAL. The setting is only a transport priority hint;
// failure must not bring down the DSH host or the local evolution service.
const net = require("node:net");
const original = net.Socket.prototype.setTypeOfService;

if (typeof original === "function" && !original.__ecologyNoTosCompat) {
  function setTypeOfServiceCompat(value) {
    try {
      return original.call(this, value);
    } catch (error) {
      if (error && error.code === "EINVAL") return this;
      throw error;
    }
  }
  setTypeOfServiceCompat.__ecologyNoTosCompat = true;
  net.Socket.prototype.setTypeOfService = setTypeOfServiceCompat;
}
