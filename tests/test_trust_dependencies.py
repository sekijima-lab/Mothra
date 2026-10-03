"""HTTP/TLS and IDNA compatibility; no external network, credentials or model files."""
import json
from pathlib import Path
import ssl
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import certifi
import idna
import requests

DOMAINS = {
    'example.org': 'example.org',
    'bücher.de': 'xn--bcher-kva.de',
    'faß.de': 'xn--fa-hia.de',
    '日本語.jp': 'xn--wgv71a119e.jp',
    'مثال.إختبار': 'xn--mgbh0fb.xn--kgbechtv',
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/json')
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'message':'分子', 'ok':True}).encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)


class TrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p = Path(cls.tmp.name)
        cls.cert = p/'localhost.pem'
        key = p/'localhost.key'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes',
                        '-keyout',str(key),'-out',str(cls.cert),'-days','1',
                        '-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cls.cert, key)
        cls.server.socket = context.wrap_socket(cls.server.socket, server_side=True)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'https://localhost:'+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.tmp.cleanup()

    def session(self):
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        return session

    def test_unicode_domains_and_requests_url(self):
        for text, ascii_name in DOMAINS.items():
            with self.subTest(domain=text):
                self.assertEqual(idna.encode(text).decode(), ascii_name)
                self.assertEqual(idna.decode(ascii_name), text)
                req = requests.Request('GET','https://'+text+'/resource').prepare()
                self.assertEqual(req.url,'https://'+ascii_name+'/resource')

    def test_invalid_domains(self):
        for domain in ['a'*64+'.org', '-bad.org', 'bad-.org', 'a..org', '\u202e.org']:
            with self.subTest(domain=domain):
                with self.assertRaises(idna.IDNAError):
                    idna.encode(domain)

    def test_tls_get_redirect_and_post_with_explicit_ca(self):
        session = self.session()
        response = session.get(self.url+'/redirect', verify=str(self.cert), timeout=5)
        self.assertEqual(response.json(), {'message':'分子','ok':True})
        self.assertEqual([r.status_code for r in response.history],[302])
        data={'smiles':'CCO','name':'分子'}
        response = session.post(self.url+'/json',json=data,verify=str(self.cert),timeout=5)
        self.assertEqual(response.json(),data)

    def test_default_ca_rejects_untrusted_tls(self):
        session = self.session()
        with self.assertRaises(requests.exceptions.SSLError):
            session.get(self.url+'/json',timeout=5)
        context=ssl.create_default_context(cafile=certifi.where())
        self.assertGreater(len(context.get_ca_certs()),100)

    def test_existing_auth_transport_compatibility(self):
        from google.auth.credentials import AnonymousCredentials
        from google.auth.transport.requests import AuthorizedSession
        from requests_oauthlib import OAuth2Session
        from google_auth_oauthlib.flow import Flow
        session=AuthorizedSession(AnonymousCredentials())
        self.addCleanup(session.close)
        session.trust_env=False
        self.assertEqual(session.get(self.url+'/json',verify=str(self.cert),timeout=5).json()['ok'],True)
        oauth=OAuth2Session(client_id='fixture',state='fixture-state')
        self.addCleanup(oauth.close)
        url,state=oauth.authorization_url('https://example.org/authorize')
        self.assertEqual(state,'fixture-state')
        self.assertEqual(parse_qs(urlparse(url).query)['client_id'],['fixture'])

    def test_distrusted_roots_removed(self):
        # This is the intended security change, not a compatibility invariant.
        context=ssl.create_default_context(cafile=certifi.where())
        names=[str(c['subject']).lower() for c in context.get_ca_certs()]
        for name in ['trustcor','e-tugra','globaltrust']:
            self.assertFalse(any(name in c for c in names), name)


if __name__ == '__main__':
    unittest.main()
