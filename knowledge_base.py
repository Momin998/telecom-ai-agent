

ISP_KNOWLEDGE_BASE = {
    "Internet": [
        {
            "id": "ISP-INT-01",
            "trigger": ["los light red", "fiber cable toot", "red light", "fiber line damage", "los indicator", "physical damage"],
            "solution_urdu": "1. Apne Router/ONT par 'LOS' light check karein. Red blink ka matlab hai fiber cable cut gayi hai ya bend hai.\n2. Router ko off karein aur peeli (yellow) fiber patch cord ko check karein ke wo murri hui toh nahi.\n3. Agar cable theek lag rahi hai, toh iska matlab bahar se line cut hai.\n4. Kripya helpline par call kar ke 'Fiber Cut' ki complaint darj karwayen taake technician physical line theek kar sake.",
            "solution_english": "1. Check the 'LOS' indicator on your Router/ONT. A flashing red light indicates a physical fiber cut or severe bend.\n2. Power off the router and inspect the yellow fiber patch cord for any sharp bends or physical damage.\n3. If the cord appears fine, the optical fiber line may be damaged externally.\n4. Please contact our helpline to log a 'Fiber Cut' ticket for a technician visit."
        },
        {
            "id": "ISP-INT-02",
            "trigger": ["ping", "gaming", "lag", "latency", "packet loss", "high ping", "delay"],
            "solution_urdu": "1. Gaming (PUBG/Valorant) mein high ping aam taur par background downloads ya wifi interference ki wajah se hoti hai.\n2. Behtareen gaming experience aur 0% packet loss ke liye hamesha LAN (Ethernet) cable use karein.\n3. Peak hours (shaam 6 se 11 baje) mein network routing delay ho sakta hai.\n4. Apne router ko 2 minute ke liye restart karein taake naya IP session connect ho.",
            "solution_english": "1. High latency (ping) and packet loss in gaming are often caused by background downloads or Wi-Fi channel interference.\n2. For an optimal gaming experience with 0% packet loss, always use a LAN (Ethernet) connection instead of Wi-Fi.\n3. Network routing delays may occasionally occur during peak traffic hours (6 PM to 11 PM).\n4. Restart your router for 2 minutes to establish a fresh IP session."
        },
        {
            "id": "ISP-INT-03",
            "trigger": ["5ghz", "show nahi ho raha", "wifi range", "signal weak", "coverage", "signal drop"],
            "solution_urdu": "1. 5GHz network ki speed bohat tez hoti hai lekin iski range kam hoti hai (sirf usi kamre tak).\n2. Agar aap router se door hain, toh hamesha 2.4GHz wale network se connect karein jiski coverage poore ghar mein hoti hai.\n3. Agar 5GHz bilkul show nahi ho raha, toh apne phone ki settings check karein (kuch purane phones 5GHz support nahi karte).\n4. Router ko TV ya microwave se door kisi khuli aur unchi jagah par rakhein.",
            "solution_english": "1. The 5GHz frequency band offers maximum bandwidth but has a significantly shorter range.\n2. If you are further away from the router, connect to the 2.4GHz network for better coverage and stability.\n3. If the 5GHz SSID is completely invisible, verify if your specific mobile device supports dual-band Wi-Fi.\n4. Ensure the router is placed in an elevated, open area away from electronics like microwaves."
        },
        {
            "id": "ISP-INT-04",
            "trigger": ["pon light blink", "internet dead", "fluctuate", "unstable", "intermittent", "disconnects"],
            "solution_urdu": "1. Agar PON light blink kar rahi hai aur rukk nahi rahi, toh iska matlab router peeche backend exchange se sync nahi ho paa raha.\n2. Router ki power cable nikal kar 5 minute wait karein aur dobara lagayen.\n3. Agar internet bar bar disconnect ho raha hai (unstable hai), toh iski wajah line mein optical power (dbm) ka kam hona ho sakti hai.\n4. Is issue ko resolve karne ke liye backend se port refresh karwani paregi, helpline par rabta karein.",
            "solution_english": "1. A continuously blinking PON light indicates that your router is failing to synchronize with the backend optical exchange.\n2. Please perform a hard reboot by unplugging the router's power cable for 5 minutes.\n3. Frequent disconnections or intermittent service are often caused by weak optical power (dbm) levels in the fiber line.\n4. Please contact support to have your backend port refreshed and optical levels diagnosed."
        },
        {
            "id": "ISP-INT-05",
            "trigger": ["cctv", "port forward", "static ip", "nat type", "server host"],
            "solution_urdu": "1. CCTV cameras ya gaming servers ko bahar se access karne ke liye Public/Static IP ki zaroorat hoti hai.\n2. Jab aap Static IP activate karwa lein, toh router ke admin panel mein ja kar 'Forwarding' ya 'Virtual Server' ke tab mein required ports open karein.\n3. Agar Double NAT ka error aa raha hai, toh ensure karein ke secondary router Bridge mode par set hai.\n4. Static IP assign karwane ke liye hamari billing team se rabta karein.",
            "solution_english": "1. To access CCTV cameras remotely or host gaming servers, a Public/Static IP assignment is strictly required.\n2. Once the Static IP is active, access the router's admin panel and configure the required open ports under the 'Forwarding' or 'Virtual Server' tab.\n3. If encountering a Double NAT error, ensure any secondary routers are configured in Bridge Mode.\n4. Please contact our billing department to subscribe to a Static IP."
        },
        {
            "id": "ISP-INT-06",
            "trigger": ["vpn", "work from home", "office network", "proxy", "cannot connect vpn"],
            "solution_urdu": "1. Agar apka corporate VPN connect nahi ho raha, toh check karein ke VPN ka protocol (PPTP/L2TP/IPsec) router ke firewall se block toh nahi.\n2. Router ke admin panel mein 'Security' ya 'ALG' setting mein ja kar VPN Passthrough ko enable karein.\n3. Baaz auqat office network hamare dynamic IPs ko block kar deta hai, is surat mein apko Static IP leni paregi.\n4. Mazeed technical rahnumai ke liye apni company ke IT department se ports ki details le kar hamein call karein.",
            "solution_english": "1. If your corporate VPN fails to connect, ensure that the VPN protocol (PPTP/L2TP/IPsec) is not blocked by the router's firewall.\n2. Access the router's admin panel, navigate to 'Security' or 'ALG' settings, and enable VPN Passthrough.\n3. Occasionally, corporate IT networks block dynamic IPs; in such cases, subscribing to a Static IP is required.\n4. For further assistance, acquire the required port details from your IT department and contact our technical team."
        }
    ],
    "Billing": [
        {
            "id": "ISP-BIL-01",
            "trigger": ["bill pay kar diya", "payment show nahi ho rahi", "service suspended", "payment not reflecting", "clear hai phir bhi net band"],
            "solution_urdu": "1. Bank apps ya Easypaisa/Jazzcash se bill jama karwane ke baad system update hone mein 30 se 60 minute lag sakte hain.\n2. Agar 1 ghanta guzar gaya hai, toh apne router ko restart karein taake apka connection active ho jaye.\n3. Agar payment ke bawajood net nahi chala, toh payment ka screenshot hamari email ya WhatsApp support par bhejein.\n4. Aap apni payment ka status hamari Customer App mein bhi check kar sakte hain.",
            "solution_english": "1. Payments made via third-party banking apps or digital wallets may take 30 to 60 minutes to reflect in our billing system.\n2. If an hour has passed since payment, please restart your router to re-authenticate your active session.\n3. If the service remains suspended, please share your successful payment screenshot with our support team.\n4. You can also verify your updated payment status directly through our official Customer App."
        },
        {
            "id": "ISP-BIL-02",
            "trigger": ["static ip charge", "extra charges", "zyada aya", "tax", "discrepancy", "incorrect billing"],
            "solution_urdu": "1. FBR ke rules ke mutabiq internet invoice par 15% Advance Income Tax aur Provincial Sales Tax lagoo hota hai.\n2. Agar aap ne Static IP lagwayi hui hai (CCTV ya Gaming ke liye), toh uska Rs. 500 monthly charge bill mein alag se add hota hai.\n3. Agar pichle mahine ka bill late pay kiya tha, toh is dafa Late Payment Surcharge (LPS) shamil hoga.\n4. Apne bill ki mukammal tafseel aur tax breakdown ke liye email par aayi hui aakhri invoice check karein.",
            "solution_english": "1. As per government regulations, a 15% Advance Income Tax and applicable Provincial Sales Taxes are levied on broadband services.\n2. If you are subscribed to a Static IP (for CCTV or hosting), a standard monthly recurring charge is applied to your invoice.\n3. A Late Payment Surcharge (LPS) is included if the previous month's invoice was cleared after the due date.\n4. Please review your latest emailed invoice for a complete breakdown of all applied taxes and extra charges."
        },
        {
            "id": "ISP-BIL-03",
            "trigger": ["downgrade", "upgrade", "package change", "pro rata", "speed barhani"],
            "solution_urdu": "1. Aap kisi bhi waqt apna internet package upgrade (speed barha) sakte hain. Iske charges current mahine ke baqi dinon ke hisaab (pro-rata) se lagte hain.\n2. Package downgrade (speed kam karna) hamesha aagay anay wale naye mahine ki 1 tareekh se lagu hota hai.\n3. Plan tabdeel karne ke liye aap hamari Customer App use kar sakte hain ya helpline par request de sakte hain.\n4. Upgrade request process hone ke baad naye speed profile ke liye router ko lazmi restart karein.",
            "solution_english": "1. You can upgrade your internet bandwidth at any time. The billing for the upgrade will be calculated on a pro-rata basis for the remaining days of the month.\n2. Package downgrades can only be scheduled to take effect from the 1st day of the upcoming billing cycle.\n3. To modify your subscription plan, utilize the official Customer App or submit a request via our helpline.\n4. Ensure you reboot your router after an upgrade to synchronize the new speed profile."
        },
        {
            "id": "ISP-BIL-04",
            "trigger": ["fup", "data cap", "limit over", "volume exhausted", "speed limit"],
            "solution_urdu": "1. Unlimited packages par bhi PTA aur Fair Usage Policy (FUP) ke tehat ek specific monthly data limit hoti hai (maslan 1TB ya 2TB).\n2. Agar aap limit cross kar lete hain, toh aapki speed temporary taur par aadhi (half) kar di jati hai.\n3. Agle mahine ki 1 tareekh ko aapka volume aur speed dobara 100% par restore ho jayegi.\n4. Apni data usage check karne ke liye apne user portal par login karein.",
            "solution_english": "1. In compliance with the Fair Usage Policy (FUP) and PTA guidelines, 'Unlimited' packages carry a maximum data cap threshold (e.g., 1TB or 2TB).\n2. Exceeding this designated volume threshold will result in a temporary reduction of your bandwidth speed.\n3. Your full volume allocation and standard speed profile will be automatically restored on the 1st of the next billing cycle.\n4. To monitor your exact bandwidth consumption, please log in to your official user portal."
        }
    ],
    "Customer Care Call": [
        {
            "id": "ISP-CC-01",
            "trigger": ["router jal gaya", "replace karwana hai", "ont device issue", "overheating", "malfunction", "hardware"],
            "solution_urdu": "1. Agar bijli ke jhatkay (voltage fluctuation) ya short circuit ki wajah se router jal gaya hai, toh ye company ki warranty mein cover nahi hota.\n2. Naye router ya power adapter ki qeemat aapke agle mahine ke bill mein installment mein add ki ja sakti hai.\n3. Agar router mein manufacturing fault hai, toh wo free replace kiya jayega.\n4. Naya router lagwane ke liye helpline par call kar ke Technician visit schedule karwayen.",
            "solution_english": "1. Hardware damage caused by voltage fluctuations, power surges, or short circuits is strictly not covered under the standard warranty.\n2. The cost of a replacement router or power adapter can be adjusted into your upcoming monthly invoice.\n3. If the hardware failure is due to a manufacturing defect, it will be replaced entirely free of charge.\n4. To request a hardware replacement, please call our helpline to schedule an official technician visit."
        },
        {
            "id": "ISP-CC-02",
            "trigger": ["ghar shift", "relocate", "connection transfer", "location", "new residence"],
            "solution_urdu": "1. Apna connection naye ghar shift karwane ke liye apko 'Relocation Request' darj karni hogi.\n2. Relocation se pehle hamari team check karegi ke naye area mein hamari fiber coverage mojood hai ya nahi.\n3. Relocation ke standard charges apply honge aur is process mein 3 se 5 working days lag sakte hain.\n4. Request darj karwane ke liye apna Customer ID aur naya mukammal address helpline par likhwayen.",
            "solution_english": "1. To transfer your internet connection to a new residence, you must submit a formal 'Relocation Request'.\n2. Prior to relocation, our survey team will verify if fiber optic coverage is available at your new designated address.\n3. Standard relocation charges apply, and the physical shifting process generally takes 3 to 5 working days.\n4. Please provide your Customer ID and complete new address to our support agent to initiate the relocation protocol."
        },
        {
            "id": "ISP-CC-03",
            "trigger": ["password change", "wifi name change", "admin panel", "configure", "setup"],
            "solution_urdu": "1. Apna WiFi naam ya password change karne ke liye browser mein router ka IP address (jaise 192.168.1.1 ya 192.168.100.1) enter karein.\n2. Router ke peeche likha hua Admin Username (jaise telecomadmin) aur Password enter karein.\n3. 'WLAN' ya 'Wireless' setting mein ja kar apna naya SSID (Naam) aur Password set kar ke save karein.\n4. Agar admin panel open nahi ho raha, toh router ke peeche diye gaye Reset button ko pin se 10 second daba kar factory reset karein.",
            "solution_english": "1. To change your Wi-Fi credentials, access your router's gateway IP (e.g., 192.168.1.1 or 192.168.100.1) via a web browser.\n2. Log in using the default Administrator Username and Password printed on the back sticker of your router.\n3. Navigate to the 'WLAN' or 'Wireless Configuration' tab to update your SSID (Network Name) and Security Key.\n4. If the admin panel is inaccessible, perform a factory reset by holding the physical Reset button on the router for 10 seconds."
        },
        {
            "id": "ISP-CC-04",
            "trigger": ["extender", "mesh", "range increase", "repeater setup", "dead zones"],
            "solution_urdu": "1. Agar bade ghar mein WiFi ke dead zones hain, toh aapko WiFi Extender ya Mesh Router system ki zaroorat hai.\n2. Extender ko router aur dead zone ke darmiyan lagayen jahan router ke kam az kam 2 signals aa rahe hon.\n3. Optimal performance ke liye, extender ko wirelessly connect karne ke bajaye LAN cable se main router ke sath connect karein.\n4. Hamari technical team se official Mesh Network lagwane ke liye helpline par request darj karwayen.",
            "solution_english": "1. If you are experiencing Wi-Fi dead zones in a large residence, deploying a Wi-Fi Extender or a Mesh Router system is highly recommended.\n2. Place the extender exactly halfway between the main router and the dead zone, ensuring it receives at least 2 bars of signal strength.\n3. For optimal throughput and zero loss, wire the extender directly to the main router using an Ethernet cable rather than connecting wirelessly.\n4. To request an official Mesh Network installation by our technical team, please register a request via the helpline."
        }
    ]
}