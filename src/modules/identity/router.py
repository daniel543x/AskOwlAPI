from fastapi import APIRouter

# Import submodule
from .auth.router import router as AuthRouter
from .role.router import router as RoleRouter
from .user.router import router as UserRouter

router = APIRouter()

# Include submodule
router.include_router(UserRouter)
router.include_router(RoleRouter)
router.include_router(AuthRouter)
