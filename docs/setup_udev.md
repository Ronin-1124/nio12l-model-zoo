# Setup APUSYS device permission

Neuron Runtime needs access to APUSYS device nodes:

```bash
ls -l /dev/apusys*
```

Install rule:

```bash
sudo tee /etc/udev/rules.d/99-mtk-apusys.rules > /dev/null <<'RULE'
# MediaTek APUSYS device permissions for Neuron SDK
KERNEL=="apusys*", MODE="0666"
RULE

sudo udevadm control --reload-rules
sudo udevadm trigger
```

