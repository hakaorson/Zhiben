Git config:
#需要在“控制面板\用户帐户\凭据管理器”删除已经存在的凭据
git config --global user.name "hakaorson"
git config --global user.email "gp18817878806@outlook.com"

Emvironments config:
Host sensetime_desktop
    HostName 10.151.113.239
    User user

Host dmb_jump
    HostName 100.64.137.141
    port 3000
    User pangao
    PasswordAuthentication no

Host dmb_target
    HostName 172.20.0.12
    Port 22
    User root
    ProxyCommand ssh -q -W %h:%p dmb_jump