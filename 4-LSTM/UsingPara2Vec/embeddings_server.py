from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler

from test_embeddings import get_embedding
import gensim

para_model = gensim.models.doc2vec.Doc2Vec.load('data/Models/paragraph_DM.doc2vec')
print("Model Loaded")
def embed(filename):
	print(filename)
	return get_embedding(filename, para_model)

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server
server = SimpleXMLRPCServer(("10.5.18.109", 11000),
                            requestHandler=RequestHandler, allow_none=True)
server.register_introspection_functions()

server.register_function(embed)

print("Server running at 6666")

# Run the server's main loop
server.serve_forever()

