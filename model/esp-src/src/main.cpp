/*
 * WebSocket wire protocol (text frames, one-byte type prefix):
 *   client→server   plain UTF-8 user message
 *   server→client   't' + token_text   (one token, may include leading space)
 *   server→client   'd'                (generation done)
 *   server→client   'e' + reason       (error)
 *   server→client   'b'                (busy - another user active)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <DNSServer.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

/* model.c is #included as a single TU; excluded from direct build via src_filter */
extern "C" {
#include "model.c"
}

/* LED pin and polarity set per-board via build flags in platformio.ini */
#ifndef LED_PIN
#  define LED_PIN        8
#endif
#ifndef LED_ACTIVE_LOW
#  define LED_ACTIVE_LOW 1
#endif
#define LED_ON  (LED_ACTIVE_LOW ? LOW  : HIGH)
#define LED_OFF (LED_ACTIVE_LOW ? HIGH : LOW)

#ifdef FAN_PIN
static inline void fan_set(bool on) { digitalWrite(FAN_PIN, on ? HIGH : LOW); }
#else
static inline void fan_set(bool) {}
#endif

static constexpr char     AP_SSID[]  = "esp-lm";
static constexpr char     AP_PASS[]  = "";
static constexpr float    GEN_TEMP   = 0.75f;
static constexpr int      GEN_MAX    = 64;
static constexpr uint16_t DNS_PORT   = 53;
static constexpr uint16_t HTTP_PORT  = 80;

static const IPAddress LOCAL_IP(192, 168, 4, 1);
static const IPAddress GATEWAY  (192, 168, 4, 1);
static const IPAddress SUBNET   (255, 255, 255, 0);

static const char HTML[] PROGMEM = R"HTML(<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>esp&#8209;lm</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font:15px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#111;color:#e8e8e8;display:flex;flex-direction:column;height:100dvh}
#log{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;
     gap:8px;-webkit-overflow-scrolling:touch}
.msg{max-width:82%;padding:9px 13px;border-radius:18px;line-height:1.5;
     word-break:break-word;white-space:pre-wrap}
.user{align-self:flex-end;background:#1d6ef5;color:#fff;border-bottom-right-radius:3px}
.bot{align-self:flex-start;background:#252525;border-bottom-left-radius:3px}
.caret{display:inline-block;width:2px;height:1em;background:#bbb;
       margin-left:2px;vertical-align:text-bottom;animation:blink .7s step-end infinite}
@keyframes blink{50%{opacity:0}}
#bar{display:flex;align-items:center;gap:8px;padding:10px 12px;
     padding-bottom:max(10px,env(safe-area-inset-bottom));
     background:#1a1a1a;border-top:1px solid #252525}
#inp{flex:1;padding:9px 14px;border-radius:20px;border:1px solid #383838;
     background:#252525;color:#e8e8e8;font-size:16px;outline:none;-webkit-appearance:none}
#inp::placeholder{color:#555}
#inp:disabled{opacity:.4}
#btn{padding:9px 18px;border-radius:20px;border:none;background:#1d6ef5;
     color:#fff;font-size:15px;font-weight:600;cursor:pointer;flex-shrink:0}
#btn:disabled{opacity:.4;cursor:not-allowed}
#st{text-align:center;font-size:12px;color:#555;padding:2px 0;min-height:16px}
</style></head><body>
<div id="log"></div>
<div id="st"></div>
<div id="bar">
  <input id="inp" type="text" placeholder="Message&hellip;" autocomplete="off"
         autocorrect="on" autocapitalize="sentences" enterkeyhint="send">
  <button id="btn">Send</button>
</div>
<script>
var log=document.getElementById('log'),
    inp=document.getElementById('inp'),
    btn=document.getElementById('btn'),
    st =document.getElementById('st');
var ws=null,botDiv=null,caret=null,busy=false;

function nearBottom(){return log.scrollHeight-log.scrollTop-log.clientHeight<100}
function scrollDown(){log.scrollTop=log.scrollHeight}
function status(msg){st.textContent=msg}
function setUI(on){inp.disabled=!on;btn.disabled=!on}

function addMsg(text,cls){
  var d=document.createElement('div');
  d.className='msg '+cls;
  if(text)d.textContent=text;
  log.appendChild(d);
  scrollDown();
  return d;
}

function startBot(){
  botDiv=addMsg('','bot');
  caret=document.createElement('span');
  caret.className='caret';
  botDiv.appendChild(caret);
}

function appendTok(text){
  if(!botDiv)startBot();
  var wasNear=nearBottom();
  botDiv.insertBefore(document.createTextNode(text),caret);
  if(wasNear)scrollDown();
}

function doneBot(err){
  if(caret&&botDiv)botDiv.removeChild(caret);
  if(err&&botDiv)botDiv.textContent='[error]';
  caret=null;botDiv=null;busy=false;
  setUI(true);inp.focus();
}

function connect(){
  if(ws&&ws.readyState<=1)return;
  status('Connecting…');setUI(false);
  ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onopen=function(){status('');setUI(true);inp.focus()};
  ws.onclose=function(){
    setUI(false);status('Reconnecting…');
    doneBot(false);
    setTimeout(connect,2000);
  };
  ws.onerror=function(){ws.close()};
  ws.onmessage=function(e){
    var msg=String(e.data),type=msg.charAt(0),body=msg.slice(1);
    if(type==='t'){appendTok(body);}
    else if(type==='d'){doneBot(false);}
    else if(type==='e'){doneBot(true);}
    else if(type==='b'){status('Busy - try again shortly');setTimeout(function(){status('')},2500);}
  };
}

function send(){
  var text=inp.value.trim();
  if(!text||!ws||ws.readyState!==1||busy)return;
  inp.value='';busy=true;
  addMsg(text,'user');
  startBot();
  setUI(false);
  ws.send(text);
}

btn.addEventListener('click',send);
inp.addEventListener('keydown',function(e){if(e.key==='Enter'&&!inp.disabled)send()});

/* prevent double-tap zoom on iOS */
var lastTap=0;
document.addEventListener('touchend',function(e){
  var n=Date.now();if(n-lastTap<350)e.preventDefault();lastTap=n;
},{passive:false});

connect();
</script></body></html>)HTML";

static DNSServer      dnsServer;
static AsyncWebServer httpServer(HTTP_PORT);
static AsyncWebSocket ws("/ws");

static LMState        lm_state;

/* All volatile: written/read from both WiFi-task and gen_task */
static volatile bool     busy       = false;
static volatile bool     has_client = false;
static volatile uint32_t active_id  = 0;

/* Generation task args - only written when busy==false, safe to copy at task start */
static struct {
    char     prompt[256];
    uint32_t client_id;
} gen_arg;

static void gen_task(void *) {
    /* Copy args immediately before any yield point */
    char     prompt[256];
    uint32_t cid = gen_arg.client_id;
    memcpy(prompt, gen_arg.prompt, sizeof(prompt));

    /* Protocol buffer: type byte + token text (max ~20 UTF-8 bytes) + NUL */
    char msg[32];

    fan_set(true);
    lm_prime(prompt, &lm_state);

    bool first = true;
    for (int i = 0; i < GEN_MAX; i++) {
        if (!has_client) break;

        /* First token: mask SEP/EOS so response is never empty */
        int tok = (i == 0) ? lm_next_force(&lm_state, GEN_TEMP)
                           : lm_next(&lm_state, GEN_TEMP);
        if (tok == SEP_ID || tok == EOS_ID) break;

        msg[0] = 't';
        int tlen = tok_decode_one(tok, !first, msg + 1, (int)sizeof(msg) - 1);
        first = false;

        if (tlen > 0) {
            ws.text(cid, msg);
        }

        vTaskDelay(1);            /* yield to WiFi/lwIP every token */
    }

    fan_set(false);

    if (has_client) {
        msg[0] = 'd'; msg[1] = '\0';
        ws.text(cid, msg);
    }

    busy = false;
    vTaskDelete(nullptr);
}

static void onWsEvent(AsyncWebSocket       *server,
                      AsyncWebSocketClient *client,
                      AwsEventType          type,
                      void                 *arg,
                      uint8_t              *data,
                      size_t                len)
{
    (void)server;

    if (type == WS_EVT_CONNECT) {
        if (has_client) {
            client->text("b");
            client->close();
            return;
        }
        has_client = true;
        active_id  = client->id();
        lm_init(&lm_state);
        Serial.printf("[ws] connect  id=%u\n", active_id);

    } else if (type == WS_EVT_DISCONNECT) {
        if (client->id() == active_id) {
            has_client = false;
            busy       = false;
            fan_set(false);
            lm_init(&lm_state);
            Serial.printf("[ws] disconnect  id=%u\n", client->id());
        }

    } else if (type == WS_EVT_DATA) {
        AwsFrameInfo *info = (AwsFrameInfo *)arg;
        /* Only handle complete, single-frame text messages */
        if (!info->final || info->index != 0 || info->opcode != WS_TEXT) return;

        if (busy) {
            client->text("b");
            return;
        }
        if (len == 0) return;

        busy = true;

        size_t plen = len < sizeof(gen_arg.prompt) - 1 ? len
                                                        : sizeof(gen_arg.prompt) - 1;
        memcpy(gen_arg.prompt, data, plen);
        gen_arg.prompt[plen] = '\0';
        gen_arg.client_id    = client->id();

        /* Low-priority task so WiFi/TCP can preempt between tokens */
        xTaskCreate(gen_task, "gen", 8192, nullptr, 1, nullptr);

    } else if (type == WS_EVT_ERROR) {
        Serial.printf("[ws] error  id=%u\n", client->id());
    }
}

static void serveChat(AsyncWebServerRequest *req) {
    req->send(200, "text/html", HTML);
}

static void redirectHome(AsyncWebServerRequest *req) {
    /* 302 so browsers re-check - captive portal sniffers follow it */
    req->redirect("http://192.168.4.1/");
}

void setup() {
    Serial.begin(115200);
    delay(300);

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LED_OFF);
#ifdef FAN_PIN
    pinMode(FAN_PIN, OUTPUT);
    fan_set(false);
#endif

    srand((unsigned)esp_random());

    WiFi.mode(WIFI_AP);
    WiFi.softAPConfig(LOCAL_IP, GATEWAY, SUBNET);
    WiFi.softAP(AP_SSID, *AP_PASS ? AP_PASS : nullptr);
    Serial.printf("[wifi] AP: %s  IP: %s\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str());

    dnsServer.setErrorReplyCode(DNSReplyCode::NoError);
    dnsServer.start(DNS_PORT, "*", LOCAL_IP);

    ws.onEvent(onWsEvent);
    httpServer.addHandler(&ws);

    httpServer.on("/", HTTP_GET, serveChat);
    httpServer.on("/generate_204",              HTTP_GET, redirectHome);
    httpServer.on("/gen_204",                   HTTP_GET, redirectHome);
    httpServer.on("/hotspot-detect.html",       HTTP_GET, serveChat);
    httpServer.on("/library/test/success.html", HTTP_GET, redirectHome);
    httpServer.on("/connecttest.txt",           HTTP_GET, redirectHome);
    httpServer.on("/ncsi.txt",                  HTTP_GET, [](AsyncWebServerRequest *r){
        r->send(200, "text/plain", "Microsoft NCSI");
    });
    httpServer.on("/canonical.html",            HTTP_GET, redirectHome);

    httpServer.onNotFound(redirectHome);

    httpServer.begin();
    Serial.println("[http] server started");

    lm_init(&lm_state);
    Serial.println("[lm] model ready");
}

void loop() {
    dnsServer.processNextRequest();
    ws.cleanupClients();

    /* LED status (active-low):
     *   no client   → slow blink  1 Hz  (waiting)
     *   client idle → solid on          (connected)
     *   generating  → fast blink  8 Hz  (thinking)  */
    static uint32_t ledLast = 0;
    static bool     ledState = false;
    uint32_t now = millis();
    uint32_t interval = has_client ? (busy ? 62 : 0) : 500;

    if (interval == 0) {
        digitalWrite(LED_PIN, LED_ON);
    } else if (now - ledLast >= interval) {
        ledLast  = now;
        ledState = !ledState;
        digitalWrite(LED_PIN, ledState ? LED_ON : LED_OFF);
    }

    delay(5);
}
