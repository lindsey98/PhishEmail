# On Ubuntu, install rspamd
```commandline
sudo apt-get install -y lsb-release wget gpg  # for install
CODENAME=`lsb_release -c -s`
sudo mkdir -p /etc/apt/keyrings
wget -O- https://rspamd.com/apt-stable/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rspamd.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/rspamd.gpg] http://rspamd.com/apt-stable/ $CODENAME main" | sudo tee /etc/apt/sources.list.d/rspamd.list
echo "deb-src [signed-by=/etc/apt/keyrings/rspamd.gpg] http://rspamd.com/apt-stable/ $CODENAME main"  | sudo tee -a /etc/apt/sources.list.d/rspamd.list
sudo apt-get update
sudo apt-get --no-install-recommends install rspamd
```

# [IMPORTANT] Please configure the /etc/rspamd/local.d/worker-controller.inc properly
```commandline
sudo vim /etc/rspamd/local.d/worker-controller.inc
```

Make sure that it looks like this
```
bind_socket = "localhost:11334";
```

Restart rspamd
```commandline
sudo systemctl restart rspamd 
```

# Change the rspamd settings, disable some unimportant checks (missing Date, missing MIME version, etc)
```commandline
cd /etc/rspamd/local.d/
sudo nano groups.conf
```

Add the following configurations in groups.conf
```commandline
symbols {
    HFILTER_HOSTNAME_UNKNOWN {
        score = 0;
    }
    MIME_HEADER_CTYPE_ONLY {
        score = 0;
    }
    MIME_HTML_ONLY {
        score = 0;
    }
    MISSING_DATE {
        score = 0;
    }
    MISSING_MID {
        score = 0;
    }
    MISSING_MIME_VERSION {
        score = 0;
    }
}
```

Restart rspamd
```commandline
 sudo systemctl restart rspamd 
```

# Verify that rspamd is active
```commandline
 sudo systemctl status rspamd
```

# Test rspamd on a single .eml file
```commandline
sudo rspamc < email_file.eml
```