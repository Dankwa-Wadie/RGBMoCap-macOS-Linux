"""Draw a CPU-skinned mesh through the real renderers in a GL 4.1 core context."""
import sys
import numpy as np
from PySide6.QtGui import QGuiApplication, QOpenGLContext, QOffscreenSurface, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
from mocap.platform_compat import GL_VERSION
from mocap.rig_gl import RigRenderer

app = QGuiApplication(sys.argv[:1])
fmt = QSurfaceFormat(); fmt.setVersion(*GL_VERSION); fmt.setProfile(QSurfaceFormat.CoreProfile); fmt.setDepthBufferSize(24)
ctx = QOpenGLContext(); ctx.setFormat(fmt); assert ctx.create()
surf = QOffscreenSurface(); surf.setFormat(fmt); surf.create(); assert ctx.makeCurrent(surf)
print('context', ctx.format().majorVersion(), ctx.format().minorVersion())
ff = QOpenGLFramebufferObjectFormat(); ff.setAttachment(QOpenGLFramebufferObject.Depth)
fbo = QOpenGLFramebufferObject(256, 256, ff); fbo.bind()

# Octahedron, one bone per vertex (rigid mode) -> exercises rig skin + draw.
verts = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], np.float32) * .5
faces = np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]], np.uint32)
packed = np.concatenate([verts, verts, np.zeros((6,1),np.float32)], 1).ravel()
asset = dict(key='oct', vertex_count=6, faces=faces, mode='rigid', bone_count=1, packed=packed,
             offsets=np.zeros(7, np.uint32), neighbors=np.zeros(2, np.uint32))
r = RigRenderer(); r.load(asset)
bone = np.eye(4, dtype=np.float32); bone[:3, 3] = [0, 0, 0]
r.deform(bone.ravel(), 'f0')
gl = r.gl
gl.glViewport(0, 0, 256, 256); gl.glClearColor(0, 0, 0, 1); gl.glClear(0x4000 | 0x100)
r.draw(np.array([0, 0, -3.]), np.eye(3), 256, 256, color=(1, .5, .2), clear_depth=False)
img = fbo.toImage()
lit = sum(1 for y in range(0, 256, 4) for x in range(0, 256, 4) if img.pixelColor(x, y).red() > 20)
print('records', r.read_vertices()[:2].round(3).tolist())
print('lit samples', lit, '/', 64 * 64)
img.save(sys.argv[1] if len(sys.argv) > 1 else 'gl41.png')
assert lit > 50, 'nothing rendered'
print('GL41 RENDER OK')
