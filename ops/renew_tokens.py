"""
Workaround for google-colab-cli issue #106 (fix pending in PR #136).

The runtime-proxy token that `colab new` stores in ~/.config/colab-cli/sessions.json
expires after 3600 s and the CLI never renews it. After ~1 hour every exec/upload/
download gets 404/401, the CLI wrongly concludes the session is gone, deletes it and
kills its keep-alive daemon - and Colab then reclaims the idle VM.

This script asks the Colab API for the current assignments (which include a FRESH
proxy token + URL for every live runtime) and writes them into the CLI's session
state. Run it every few minutes (see renew_loop.sh). Must be run with the CLI's own
Python:  ~/.local/share/uv/tools/google-colab-cli/bin/python renew_tokens.py
"""
from datetime import datetime

from colab_cli.common import state

assignments = {a.endpoint: a for a in state.client.list_assignments()}
now = datetime.now().strftime("%H:%M:%S")
for name, sess in state.store.list().items():
    a = assignments.get(sess.endpoint)
    if a is None:
        print(f"{now} {name}: not assigned on the server any more")
        continue
    info = a.runtime_proxy_info
    changed = sess.token != info.token or sess.url != info.url
    sess.token, sess.url = info.token, info.url
    state.store.add(sess)          # locked read-modify-write of sessions.json
    print(f"{now} {name}: token {'renewed' if changed else 'unchanged'} (server says valid {info.token_expires_in_seconds}s)")
