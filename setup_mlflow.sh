sudo pip3 install mlflow -y

echo "Installed mlflow"

sudo apt install nginx -y

echo "Installed nginx"

sudo apt-get install apache2-utils -y

echo "Installed apache2-utils"

cd /etc/nginx/sites-enabled

sudo htpasswd -c /etc/nginx/.htpasswd USERNAME

echo "server {
    listen 8080;
    server_name YOUR_IP_OR_DOMAIN;
    auth_basic “Administrator-Area”;
    auth_basic_user_file /etc/nginx/.htpasswd; 

    location / {
        proxy_pass http://localhost:8000;
        include /etc/nginx/proxy_params;
        proxy_redirect off;
    }
}
" >> mlflow

echo "added mlflow nginx configuration file"

sudo service nginx start

echo "started nginx"

cd /home/ubuntu

cd  /etc/systemd/system

echo "
[Unit]
Description=MLflow tracking server
After=network.target

[Service]
Restart=on-failure
RestartSec=30

ExecStart=/bin/bash -c 'PATH=/home/ubuntu/anaconda3/envs/CREATED_ENV/bin/:$PATH exec mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://wafer-mlflow/ --host 0.0.0.0 -p 8000'

[Install]
WantedBy=multi-user.target
" >> mlflow-tracking.service

echo "added service file for mlflow"

cd /home/ubuntu

sudo systemctl daemon-reload

echo "daemon reloaded"

sudo systemctl enable mlflow-tracking 

echo "enabled mlflow-tracking"

sudo systemctl start mlflow-tracking

echo "started mlflow-tracking as service"

echo "mlflow setup is done"

echo "check your ip with port 8080"
