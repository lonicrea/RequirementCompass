const PROJECTS_STORAGE_KEY = 'requirement_compass_projects'

const safeParse = (raw, fallback) => {
  try {
    return JSON.parse(raw)
  } catch {
    return fallback
  }
}

export const loadProjects = () => {
  if (typeof window === 'undefined') return []
  const parsed = safeParse(localStorage.getItem(PROJECTS_STORAGE_KEY) || '[]', [])
  return Array.isArray(parsed) ? parsed : []
}

export const saveProjects = (projects) => {
  if (typeof window === 'undefined') return
  localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(Array.isArray(projects) ? projects : []))
}

export const upsertProject = (project) => {
  if (!project?.id || typeof window === 'undefined') return
  const projects = loadProjects().filter((item) => item?.id !== project.id)
  projects.push(project)
  saveProjects(projects)
}

export const removeProject = (projectId) => {
  if (!projectId || typeof window === 'undefined') return
  saveProjects(loadProjects().filter((item) => item?.id !== projectId))
}

export { PROJECTS_STORAGE_KEY }
