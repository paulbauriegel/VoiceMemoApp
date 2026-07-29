@file:OptIn(
    androidx.compose.foundation.ExperimentalFoundationApi::class,
    androidx.compose.material3.ExperimentalMaterial3Api::class,
)

package com.example.voicememo.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.DriveFileMove
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Archive
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Lightbulb
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.MoreVert
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material.icons.rounded.SelectAll
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.School
import androidx.compose.material.icons.rounded.Share
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material.icons.rounded.StarBorder
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material.icons.rounded.Work
import androidx.compose.material.icons.rounded.Unarchive
import androidx.compose.material3.Checkbox
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LargeFloatingActionButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.voicememo.data.ProjectIcon
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

private enum class RecordingsFilter {
    Unassigned,
    All,
    Starred,
}

private enum class LibraryCollectionTab {
    Projects,
    Folders,
}

@Composable
fun VoiceMemoLibraryScreen(
    state: VoiceMemoAppUiState,
    onAction: (VoiceMemoUiAction) -> Unit,
    modifier: Modifier = Modifier,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val selectionMode = state.selectedMemoIds.isNotEmpty()
    var showSelectionActions by rememberSaveable { mutableStateOf(false) }
    var showMoveSheet by rememberSaveable { mutableStateOf(false) }
    var projectDialogMemoIds by remember { mutableStateOf<Set<String>?>(null) }
    var recordingsFilter by rememberSaveable { mutableStateOf(RecordingsFilter.Unassigned) }
    var collectionTab by rememberSaveable { mutableStateOf(LibraryCollectionTab.Projects) }
    var projectSearchExpanded by rememberSaveable(state.activeProjectId) {
        mutableStateOf(state.query.isNotBlank())
    }
    var projectEditorOpen by rememberSaveable(state.activeProjectId) { mutableStateOf(false) }
    var projectNameDraft by rememberSaveable(state.activeProjectId) {
        mutableStateOf(state.activeProjectName.orEmpty())
    }
    var projectIconDraftName by rememberSaveable(state.activeProjectId) {
        mutableStateOf(state.activeProjectIcon.name)
    }
    var archivedProjectsExpanded by rememberSaveable { mutableStateOf(false) }
    val activeProjects = state.projects.filterNot(ProjectListItemUi::isArchived)
    val archivedProjects = state.projects.filter(ProjectListItemUi::isArchived)
    val projectIconDraft = runCatching {
        ProjectIcon.valueOf(projectIconDraftName)
    }.getOrDefault(ProjectIcon.WORK)

    val closeProjectSearch = {
        projectSearchExpanded = false
        if (state.query.isNotEmpty()) {
            onAction(VoiceMemoUiAction.SearchChanged(""))
        }
    }

    BackHandler(
        enabled = state.activeProjectId != null && projectSearchExpanded && !selectionMode,
        onBack = closeProjectSearch,
    )
    BackHandler(
        enabled = state.activeProjectId != null && projectEditorOpen && !selectionMode,
        onBack = { projectEditorOpen = false },
    )

    val displayedMemos = when {
        state.activeProjectId != null || state.query.isNotBlank() -> state.memos
        recordingsFilter == RecordingsFilter.Unassigned ->
            state.memos.filter { it.projectId == null }
        recordingsFilter == RecordingsFilter.Starred -> state.memos.filter { it.isStarred }
        else -> state.memos
    }

    Scaffold(
        modifier = modifier
            .fillMaxSize()
            .testTag("library_screen"),
        topBar = {
            if (selectionMode) {
                SelectionTopBar(
                    selectedCount = state.selectedMemoIds.size,
                    onClose = { onAction(VoiceMemoUiAction.ClearSelection) },
                    onMore = { showSelectionActions = true },
                )
            } else if (state.activeProjectId != null && projectEditorOpen) {
                ProjectEditorTopBar(
                    saveEnabled = projectNameDraft.isNotBlank(),
                    onClose = { projectEditorOpen = false },
                    onSave = {
                        onAction(
                            VoiceMemoUiAction.UpdateProject(
                                projectId = state.activeProjectId,
                                name = projectNameDraft,
                                icon = projectIconDraft,
                            ),
                        )
                        projectEditorOpen = false
                    },
                )
            } else if (state.activeProjectId != null && projectSearchExpanded) {
                ProjectSearchTopBar(
                    query = state.query,
                    onQueryChanged = { onAction(VoiceMemoUiAction.SearchChanged(it)) },
                    onClose = closeProjectSearch,
                )
            } else if (state.activeProjectId != null) {
                ProjectTopBar(
                    projectName = state.activeProjectName ?: "Project",
                    onBack = { onAction(VoiceMemoUiAction.CloseProject) },
                    onSearch = { projectSearchExpanded = true },
                    onEdit = {
                        projectNameDraft = state.activeProjectName.orEmpty()
                        projectIconDraftName = state.activeProjectIcon.name
                        projectEditorOpen = true
                    },
                )
            } else {
                LibrarySearchTopBar(
                    query = state.query,
                    onQueryChanged = { onAction(VoiceMemoUiAction.SearchChanged(it)) },
                    onOpenSettings = { onAction(VoiceMemoUiAction.OpenSettings) },
                )
            }
        },
        bottomBar = {
            if (!selectionMode && !projectEditorOpen) {
                LibraryBottomBar(
                    selected = state.libraryTab,
                    recordEnabled = !state.isImportingAudio,
                    onSelected = {
                        onAction(VoiceMemoUiAction.LibraryTabChanged(it))
                    },
                    onRecord = { onAction(VoiceMemoUiAction.StartRecording) },
                )
            }
        },
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(
                start = spacing.sm,
                top = spacing.xs,
                end = spacing.sm,
                bottom = spacing.lg,
            ),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (projectEditorOpen && state.activeProjectId != null) {
                item(key = "project-editor") {
                    ProjectEditor(
                        projectName = projectNameDraft,
                        selectedIcon = projectIconDraft,
                        archived = state.activeProjectArchived,
                        onProjectNameChanged = { projectNameDraft = it.take(80) },
                        onIconSelected = { projectIconDraftName = it.name },
                        onArchiveChanged = { archived ->
                            projectEditorOpen = false
                            onAction(
                                VoiceMemoUiAction.SetProjectArchived(
                                    projectId = state.activeProjectId,
                                    archived = archived,
                                ),
                            )
                        },
                    )
                }
            } else if (
                state.libraryTab == LibraryTab.Projects &&
                state.activeProjectId == null
            ) {
                item(key = "library-collection-tabs") {
                    LibraryCollectionTabs(
                        selected = collectionTab,
                        onSelected = { collectionTab = it },
                    )
                }

                if (collectionTab == LibraryCollectionTab.Folders) {
                    item(key = "folders-empty") {
                        FoldersEmptyState()
                    }
                } else {
                    item(key = "projects-heading") {
                        MyProjectsHeading(
                            onAdd = { projectDialogMemoIds = emptySet() },
                        )
                    }
                    val visibleProjects = activeProjects.filter { project ->
                        state.query.isBlank() ||
                            project.name.contains(state.query, ignoreCase = true)
                    }
                    if (visibleProjects.isEmpty()) {
                        item(key = "projects-empty") {
                            ProjectsEmptyState(
                                hasQuery = state.query.isNotBlank(),
                                onCreate = { projectDialogMemoIds = emptySet() },
                            )
                        }
                    } else {
                        items(
                            items = visibleProjects,
                            key = { "project-${it.id}" },
                        ) { project ->
                            ProjectCard(
                                project = project,
                                accentIndex = activeProjects.indexOf(project),
                                waveform = state.memos
                                    .firstOrNull { it.projectId == project.id }
                                    ?.waveform
                                    .orEmpty(),
                                onOpen = {
                                    onAction(VoiceMemoUiAction.OpenProject(project.id))
                                },
                                onDelete = {
                                    onAction(VoiceMemoUiAction.RequestDeleteProject(project.id))
                                },
                            )
                        }
                    }
                    if (archivedProjects.isNotEmpty()) {
                        item(key = "archived-projects") {
                            ArchivedProjectsRow(
                                count = archivedProjects.size,
                                expanded = archivedProjectsExpanded,
                                onToggle = {
                                    archivedProjectsExpanded = !archivedProjectsExpanded
                                },
                            )
                        }
                        if (archivedProjectsExpanded) {
                            items(
                                items = archivedProjects,
                                key = { "archived-project-${it.id}" },
                            ) { project ->
                                ProjectCard(
                                    project = project,
                                    accentIndex = archivedProjects.indexOf(project),
                                    waveform = state.memos
                                        .firstOrNull { it.projectId == project.id }
                                        ?.waveform
                                        .orEmpty(),
                                    onOpen = {
                                        onAction(VoiceMemoUiAction.OpenProject(project.id))
                                    },
                                    onDelete = {
                                        onAction(
                                            VoiceMemoUiAction.RequestDeleteProject(project.id),
                                        )
                                    },
                                )
                            }
                        }
                    }
                }
            } else {
                if (state.activeProjectId == null) {
                    item(key = "recordings-filter") {
                        RecordingsFilterTabs(
                            selected = recordingsFilter,
                            onSelected = { recordingsFilter = it },
                        )
                    }
                }

                if (displayedMemos.isEmpty()) {
                    item(key = "empty") {
                        LibraryEmptyState(
                            hasQuery = state.query.isNotBlank(),
                            inProject = state.activeProjectId != null,
                            starred = recordingsFilter == RecordingsFilter.Starred,
                            unassigned = recordingsFilter == RecordingsFilter.Unassigned,
                        )
                    }
                } else {
                    if (state.query.isNotBlank()) {
                        item(key = "search-result-count") {
                            SectionHeading(
                                if (displayedMemos.size == 1) {
                                    "1 result"
                                } else {
                                    "${displayedMemos.size} results"
                                },
                            )
                        }
                    }
                    displayedMemos
                        .groupBy { recordingSectionLabel(it.createdAtMillis) }
                        .forEach { (section, memos) ->
                            item(key = "recording-section-$section") {
                                SectionHeading(section)
                            }
                            items(
                                items = memos,
                                key = { it.id },
                            ) { memo ->
                                VoiceMemoCard(
                                    memo = memo,
                                    selected = memo.id in state.selectedMemoIds,
                                    selectionMode = selectionMode,
                                    onOpen = {
                                        if (selectionMode) {
                                            onAction(
                                                VoiceMemoUiAction.ToggleMemoSelection(memo.id),
                                            )
                                        } else {
                                            onAction(VoiceMemoUiAction.OpenMemo(memo.id))
                                        }
                                    },
                                    onLongPress = {
                                        onAction(VoiceMemoUiAction.ToggleMemoSelection(memo.id))
                                    },
                                    onToggleSelection = {
                                        onAction(VoiceMemoUiAction.ToggleMemoSelection(memo.id))
                                    },
                                    onTogglePlayback = {
                                        onAction(VoiceMemoUiAction.ToggleMemoPlayback(memo.id))
                                    },
                                    onToggleStarred = {
                                        onAction(VoiceMemoUiAction.ToggleMemoStarred(memo.id))
                                    },
                                )
                            }
                        }
                }

                if (
                    state.activeProjectId == null &&
                    state.query.isBlank() &&
                    recordingsFilter == RecordingsFilter.All
                ) {
                    item(key = "recordings-projects-heading") {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(start = 8.dp, top = 20.dp, end = 2.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                text = "Projects",
                                style = MaterialTheme.typography.headlineSmall,
                                modifier = Modifier.weight(1f),
                            )
                            TextButton(
                                onClick = {
                                    onAction(
                                        VoiceMemoUiAction.LibraryTabChanged(LibraryTab.Projects),
                                    )
                                },
                            ) {
                                Text("See all")
                            }
                            IconButton(
                                onClick = { projectDialogMemoIds = emptySet() },
                                modifier = Modifier.testTag("new_project_action"),
                            ) {
                                Icon(Icons.Rounded.Add, contentDescription = "New project")
                            }
                        }
                    }
                    if (activeProjects.isEmpty()) {
                        item(key = "recordings-projects-empty") {
                            ProjectsEmptyState(
                                hasQuery = false,
                                onCreate = { projectDialogMemoIds = emptySet() },
                                compact = true,
                            )
                        }
                    } else {
                        items(
                            items = activeProjects.take(3),
                            key = { "recordings-project-${it.id}" },
                        ) { project ->
                            ProjectCard(
                                project = project,
                                accentIndex = activeProjects.indexOf(project),
                                waveform = state.memos
                                    .firstOrNull { it.projectId == project.id }
                                    ?.waveform
                                    .orEmpty(),
                                onOpen = {
                                    onAction(VoiceMemoUiAction.OpenProject(project.id))
                                },
                                onDelete = {
                                    onAction(
                                        VoiceMemoUiAction.RequestDeleteProject(project.id),
                                    )
                                },
                            )
                        }
                    }
                }
            }
        }
    }

    if (showSelectionActions) {
        SelectionActionsSheet(
            selectedCount = state.selectedMemoIds.size,
            onDismiss = { showSelectionActions = false },
            onDelete = {
                showSelectionActions = false
                onAction(VoiceMemoUiAction.RequestDelete(state.selectedMemoIds))
            },
            onShare = {
                showSelectionActions = false
                onAction(VoiceMemoUiAction.RequestShare(state.selectedMemoIds))
            },
            onSelectAll = {
                showSelectionActions = false
                onAction(VoiceMemoUiAction.SelectAll)
            },
            onMove = {
                showSelectionActions = false
                showMoveSheet = true
            },
        )
    }

    if (showMoveSheet) {
        MoveToProjectSheet(
            projects = activeProjects,
            onDismiss = { showMoveSheet = false },
            onSelect = { projectId ->
                showMoveSheet = false
                onAction(
                    VoiceMemoUiAction.AssignMemosToProject(
                        memoIds = state.selectedMemoIds,
                        projectId = projectId,
                    ),
                )
            },
            onCreate = {
                showMoveSheet = false
                projectDialogMemoIds = state.selectedMemoIds
            },
        )
    }

    projectDialogMemoIds?.let { memoIds ->
        CreateProjectDialog(
            onDismiss = { projectDialogMemoIds = null },
            onCreate = { name ->
                projectDialogMemoIds = null
                onAction(VoiceMemoUiAction.CreateProject(name, memoIds))
            },
        )
    }
}

@Composable
private fun LibrarySearchTopBar(
    query: String,
    onQueryChanged: (String) -> Unit,
    onOpenSettings: () -> Unit,
) {
    val spacing = LocalVoiceMemoSpacing.current
    Surface(
        color = MaterialTheme.colorScheme.surface,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(
                    start = spacing.sm,
                    top = spacing.xs,
                    end = spacing.xs,
                    bottom = spacing.xs,
                )
                .testTag("library_search_header"),
            horizontalArrangement = Arrangement.spacedBy(spacing.xs),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            VoiceMemoSearchField(
                query = query,
                onQueryChanged = onQueryChanged,
                modifier = Modifier
                    .weight(1f)
                    .testTag("library_search_field"),
            )
            IconButton(
                onClick = onOpenSettings,
                modifier = Modifier
                    .size(48.dp)
                    .testTag("settings_action"),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Settings,
                    contentDescription = "Settings",
                )
            }
        }
    }
}

@Composable
private fun ProjectTopBar(
    projectName: String,
    onBack: () -> Unit,
    onSearch: () -> Unit,
    onEdit: () -> Unit,
) {
    TopAppBar(
        title = {
            Text(
                text = projectName,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.testTag("project_title"),
            )
        },
        navigationIcon = {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                    contentDescription = "Back to projects",
                )
            }
        },
        actions = {
            IconButton(
                onClick = onSearch,
                modifier = Modifier.testTag("project_search_action"),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Search,
                    contentDescription = "Search in project",
                )
            }
            IconButton(
                onClick = onEdit,
                modifier = Modifier.testTag("project_edit_action"),
            ) {
                Icon(
                    imageVector = Icons.Rounded.MoreVert,
                    contentDescription = "Edit project",
                )
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
        modifier = Modifier.testTag("project_header"),
    )
}

@Composable
private fun ProjectEditorTopBar(
    saveEnabled: Boolean,
    onClose: () -> Unit,
    onSave: () -> Unit,
) {
    TopAppBar(
        title = { Text("Edit project") },
        navigationIcon = {
            IconButton(onClick = onClose) {
                Icon(
                    imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                    contentDescription = "Close project editor",
                )
            }
        },
        actions = {
            IconButton(
                onClick = onSave,
                enabled = saveEnabled,
                modifier = Modifier.testTag("save_project_action"),
            ) {
                Icon(
                    imageVector = Icons.Rounded.Check,
                    contentDescription = "Save project",
                )
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
        modifier = Modifier.testTag("project_editor_header"),
    )
}

@Composable
private fun ProjectSearchTopBar(
    query: String,
    onQueryChanged: (String) -> Unit,
    onClose: () -> Unit,
) {
    val spacing = LocalVoiceMemoSpacing.current
    val focusRequester = remember { FocusRequester() }

    LaunchedEffect(Unit) {
        focusRequester.requestFocus()
    }

    Surface(
        color = MaterialTheme.colorScheme.surface,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.statusBars)
                .padding(
                    start = spacing.xs,
                    top = spacing.xs,
                    end = spacing.sm,
                    bottom = spacing.xs,
                )
                .testTag("project_search_header"),
            horizontalArrangement = Arrangement.spacedBy(spacing.xs),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(
                onClick = onClose,
                modifier = Modifier.size(48.dp),
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Rounded.ArrowBack,
                    contentDescription = "Close project search",
                )
            }
            VoiceMemoSearchField(
                query = query,
                onQueryChanged = onQueryChanged,
                modifier = Modifier
                    .weight(1f)
                    .focusRequester(focusRequester)
                    .testTag("project_search_field"),
            )
        }
    }
}

@Composable
private fun RecordingsFilterTabs(
    selected: RecordingsFilter,
    onSelected: (RecordingsFilter) -> Unit,
) {
    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp)
            .testTag("recordings_filter_tabs"),
    ) {
        Row(
            modifier = Modifier.padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            RecordingsFilter.entries.forEach { filter ->
                val isSelected = filter == selected
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = if (isSelected) {
                        MaterialTheme.colorScheme.secondaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceContainerHigh
                    },
                    contentColor = if (isSelected) {
                        MaterialTheme.colorScheme.onSecondaryContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    modifier = Modifier
                        .weight(1f)
                        .clip(MaterialTheme.shapes.extraLarge)
                        .clickable { onSelected(filter) }
                        .semantics {
                            role = Role.Tab
                            contentDescription = "${filter.name} recordings"
                        }
                        .testTag("recordings_filter_${filter.name.lowercase()}"),
                ) {
                    Text(
                        text = filter.name,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(vertical = 12.dp),
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                    )
                }
            }
        }
    }
}

@Composable
private fun LibraryCollectionTabs(
    selected: LibraryCollectionTab,
    onSelected: (LibraryCollectionTab) -> Unit,
) {
    Surface(
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.surfaceContainerHigh,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp)
            .testTag("library_collection_tabs"),
    ) {
        Row(
            modifier = Modifier.padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            LibraryCollectionTab.entries.forEach { tab ->
                val isSelected = tab == selected
                Surface(
                    shape = MaterialTheme.shapes.extraLarge,
                    color = if (isSelected) {
                        MaterialTheme.colorScheme.secondaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceContainerHigh
                    },
                    contentColor = if (isSelected) {
                        MaterialTheme.colorScheme.onSecondaryContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    modifier = Modifier
                        .weight(1f)
                        .clip(MaterialTheme.shapes.extraLarge)
                        .clickable { onSelected(tab) }
                        .semantics {
                            role = Role.Tab
                            contentDescription = "${tab.name} collection"
                        }
                        .testTag("library_collection_${tab.name.lowercase()}"),
                ) {
                    Text(
                        text = tab.name,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(vertical = 12.dp),
                        textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                    )
                }
            }
        }
    }
}

@Composable
private fun LibraryBottomBar(
    selected: LibraryTab,
    recordEnabled: Boolean,
    onSelected: (LibraryTab) -> Unit,
    onRecord: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(114.dp)
            .navigationBarsPadding()
            .testTag("library_bottom_bar"),
    ) {
        Surface(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .height(80.dp)
                .padding(horizontal = 8.dp),
            shape = MaterialTheme.shapes.extraLarge,
            color = MaterialTheme.colorScheme.surfaceContainer,
            tonalElevation = 3.dp,
        ) {
            Row(
                modifier = Modifier.fillMaxSize(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                BottomDestination(
                    label = "Recordings",
                    icon = Icons.Rounded.GraphicEq,
                    selected = selected == LibraryTab.Recents,
                    onClick = { onSelected(LibraryTab.Recents) },
                    modifier = Modifier.weight(1f),
                )
                Spacer(Modifier.width(94.dp))
                BottomDestination(
                    label = "Library",
                    icon = Icons.Rounded.Folder,
                    selected = selected == LibraryTab.Projects,
                    onClick = { onSelected(LibraryTab.Projects) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
        LargeFloatingActionButton(
            onClick = {
                if (recordEnabled) onRecord()
            },
            modifier = Modifier
                .align(Alignment.TopCenter)
                .offset(y = (-2).dp)
                .size(84.dp)
                .testTag("record_fab"),
            shape = CircleShape,
            containerColor = if (recordEnabled) {
                MaterialTheme.colorScheme.primary
            } else {
                MaterialTheme.colorScheme.surfaceContainerHighest
            },
            contentColor = if (recordEnabled) {
                MaterialTheme.colorScheme.onPrimary
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
            elevation = FloatingActionButtonDefaults.elevation(),
        ) {
            Icon(
                imageVector = Icons.Rounded.Mic,
                contentDescription = "Start recording",
                modifier = Modifier.size(34.dp),
            )
        }
    }
}

@Composable
private fun RowScope.BottomDestination(
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    NavigationBarItem(
        selected = selected,
        onClick = onClick,
        icon = {
            Icon(
                imageVector = icon,
                contentDescription = null,
            )
        },
        label = {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
            )
        },
        modifier = modifier
            .semantics {
                contentDescription = "$label destination"
            }
            .testTag("bottom_${label.lowercase()}"),
        alwaysShowLabel = true,
        colors = NavigationBarItemDefaults.colors(
            selectedIconColor = MaterialTheme.colorScheme.onSecondaryContainer,
            selectedTextColor = MaterialTheme.colorScheme.onSurface,
            indicatorColor = MaterialTheme.colorScheme.secondaryContainer,
            unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
            unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
        ),
    )
}

@Composable
private fun SectionHeading(
    title: String,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 8.dp, top = 12.dp, end = 4.dp, bottom = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.weight(1f),
        )
        if (actionLabel != null && onAction != null) {
            TextButton(
                onClick = onAction,
                modifier = Modifier.testTag("section_heading_action"),
            ) {
                Text(actionLabel)
            }
        }
    }
}

@Composable
private fun MyProjectsHeading(
    onAdd: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 8.dp, top = 16.dp, end = 2.dp, bottom = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "My Projects",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.weight(1f),
        )
        IconButton(
            onClick = onAdd,
            modifier = Modifier.testTag("new_project_action"),
        ) {
            Icon(Icons.Rounded.Add, contentDescription = "New project")
        }
        Row(
            modifier = Modifier.padding(start = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Recently updated",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Icon(
                Icons.Rounded.KeyboardArrowDown,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ArchivedProjectsRow(
    count: Int,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 64.dp)
            .clip(MaterialTheme.shapes.large)
            .clickable(onClick = onToggle)
            .padding(horizontal = 8.dp, vertical = 8.dp)
            .testTag("archived_projects_toggle"),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "Archived Projects ($count)",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.weight(1f),
        )
        Icon(
            imageVector = Icons.Rounded.KeyboardArrowDown,
            contentDescription = if (expanded) {
                "Collapse archived projects"
            } else {
                "Expand archived projects"
            },
            modifier = Modifier.then(
                    if (expanded) {
                        Modifier.graphicsLayer { rotationZ = 180f }
                    } else {
                        Modifier
                    },
                ),
        )
    }
}

@Composable
private fun ProjectEditor(
    projectName: String,
    selectedIcon: ProjectIcon,
    archived: Boolean,
    onProjectNameChanged: (String) -> Unit,
    onIconSelected: (ProjectIcon) -> Unit,
    onArchiveChanged: (Boolean) -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 12.dp)
            .testTag("project_editor"),
        verticalArrangement = Arrangement.spacedBy(24.dp),
    ) {
        OutlinedTextField(
            value = projectName,
            onValueChange = onProjectNameChanged,
            label = { Text("Project name") },
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .testTag("project_name_editor"),
        )
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(
                text = "Project icon",
                style = MaterialTheme.typography.titleMedium,
            )
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                maxItemsInEachRow = 4,
            ) {
                ProjectIcon.entries.forEach { icon ->
                    val isSelected = icon == selectedIcon
                    Surface(
                        onClick = { onIconSelected(icon) },
                        modifier = Modifier
                            .size(64.dp)
                            .semantics {
                                role = Role.RadioButton
                                selected = isSelected
                            }
                            .testTag("project_icon_${icon.name.lowercase()}"),
                        shape = MaterialTheme.shapes.large,
                        color = if (isSelected) {
                            MaterialTheme.colorScheme.secondaryContainer
                        } else {
                            MaterialTheme.colorScheme.surfaceContainer
                        },
                        contentColor = if (isSelected) {
                            MaterialTheme.colorScheme.onSecondaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                imageVector = icon.imageVector(),
                                contentDescription = icon.label,
                                modifier = Modifier.size(28.dp),
                            )
                        }
                    }
                }
            }
        }
        Surface(
            onClick = { onArchiveChanged(!archived) },
            modifier = Modifier
                .fillMaxWidth()
                .testTag(if (archived) "restore_project_action" else "archive_project_action"),
            shape = MaterialTheme.shapes.large,
            color = MaterialTheme.colorScheme.surfaceContainer,
            contentColor = MaterialTheme.colorScheme.onSurface,
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 14.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(
                    imageVector = if (archived) {
                        Icons.Rounded.Unarchive
                    } else {
                        Icons.Rounded.Archive
                    },
                    contentDescription = null,
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (archived) "Restore project" else "Archive project",
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        text = if (archived) {
                            "Return this project to your active projects."
                        } else {
                            "Hide it from active projects without deleting its recordings."
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun FoldersEmptyState() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 64.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Icon(
            imageVector = Icons.Rounded.FolderOpen,
            contentDescription = null,
            modifier = Modifier.size(48.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Text(
            text = "No folders yet",
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = "Folders will give you another way to organize projects.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun ProjectCard(
    project: ProjectListItemUi,
    accentIndex: Int,
    waveform: List<Float>,
    onOpen: () -> Unit,
    onDelete: () -> Unit,
) {
    val iconContainer = when (accentIndex % 3) {
        1 -> MaterialTheme.colorScheme.tertiaryContainer
        2 -> MaterialTheme.colorScheme.primaryContainer
        else -> MaterialTheme.colorScheme.secondaryContainer
    }
    val iconContent = when (accentIndex % 3) {
        1 -> MaterialTheme.colorScheme.onTertiaryContainer
        2 -> MaterialTheme.colorScheme.onPrimaryContainer
        else -> MaterialTheme.colorScheme.onSecondaryContainer
    }
    val waveformColor = when (accentIndex % 3) {
        1 -> MaterialTheme.colorScheme.tertiary
        2 -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.secondary
    }
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.extraLarge)
            .combinedClickable(
                onClick = onOpen,
                onLongClick = onDelete,
                onLongClickLabel = "Delete project",
            )
            .semantics { role = Role.Button }
            .testTag("project_card_${project.id}"),
        shape = MaterialTheme.shapes.extraLarge,
        color = MaterialTheme.colorScheme.surfaceContainer,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 13.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                shape = MaterialTheme.shapes.large,
                color = iconContainer,
                contentColor = iconContent,
            ) {
                Icon(
                    imageVector = project.icon.imageVector(),
                    contentDescription = null,
                    modifier = Modifier.padding(13.dp).size(28.dp),
                )
            }
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = project.name,
                    style = MaterialTheme.typography.titleLarge,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    text = "${project.recordingCountLabel} • ${project.updatedLabel}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                if (project.isArchived) {
                    Text(
                        text = "Archived",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
                VoiceMemoWaveform(
                    amplitudes = waveform,
                    progress = 1f,
                    description = "Project activity for ${project.name}",
                    activeColor = waveformColor,
                    inactiveColor = waveformColor.copy(alpha = 0.45f),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp),
                )
            }
            Icon(
                imageVector = Icons.Rounded.ChevronRight,
                contentDescription = null,
                modifier = Modifier.size(28.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ProjectsEmptyState(
    hasQuery: Boolean,
    onCreate: () -> Unit,
    compact: Boolean = false,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(
                horizontal = 24.dp,
                vertical = if (compact) 20.dp else 64.dp,
            ),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (!compact) {
            Icon(
                imageVector = if (hasQuery) Icons.Rounded.Info else Icons.Rounded.FolderOpen,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.primary,
            )
        }
        Text(
            text = if (hasQuery) "No matching projects" else "No projects yet",
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = if (hasQuery) {
                "Try another project name."
            } else {
                "Create a project, then move recordings into it from selection mode."
            },
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (!hasQuery) {
            TextButton(onClick = onCreate) {
                Icon(Icons.Rounded.Add, contentDescription = null)
                Text("New project")
            }
        }
    }
}

@Composable
private fun SelectionTopBar(
    selectedCount: Int,
    onClose: () -> Unit,
    onMore: () -> Unit,
) {
    TopAppBar(
        navigationIcon = {
            IconButton(onClick = onClose) {
                Icon(
                    imageVector = Icons.Rounded.Close,
                    contentDescription = "Exit selection mode",
                )
            }
        },
        title = {
            Text(
                text = "$selectedCount selected",
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        },
        actions = {
            IconButton(onClick = onMore) {
                Icon(
                    imageVector = Icons.Rounded.MoreVert,
                    contentDescription = "Selection actions",
                )
            }
        },
    )
}

@Composable
private fun VoiceMemoCard(
    memo: VoiceMemoListItemUi,
    selected: Boolean,
    selectionMode: Boolean,
    onOpen: () -> Unit,
    onLongPress: () -> Unit,
    onToggleSelection: () -> Unit,
    onTogglePlayback: () -> Unit,
    onToggleStarred: () -> Unit,
) {
    val selectedContentColor = MaterialTheme.colorScheme.onSecondaryContainer
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.extraLarge)
            .combinedClickable(
                onClick = onOpen,
                onLongClick = onLongPress,
                onLongClickLabel = "Select recording",
            )
            .semantics { role = Role.Button }
            .testTag("memo_card_${memo.id}"),
        shape = MaterialTheme.shapes.extraLarge,
        color = if (selected) {
            MaterialTheme.colorScheme.secondaryContainer
        } else {
            MaterialTheme.colorScheme.surfaceContainer
        },
        contentColor = if (selected) {
            selectedContentColor
        } else {
            MaterialTheme.colorScheme.onSurface
        },
        tonalElevation = if (selected) 3.dp else 0.dp,
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    Text(
                        text = memo.title,
                        style = MaterialTheme.typography.titleLarge,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Text(
                        text = "${memo.dateLabel}  •  ${memo.durationLabel}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (selected) {
                            selectedContentColor.copy(alpha = 0.78f)
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                if (selectionMode) {
                    Checkbox(
                        checked = selected,
                        onCheckedChange = { onToggleSelection() },
                        modifier = Modifier.size(48.dp),
                    )
                } else {
                    IconButton(
                        onClick = onToggleStarred,
                        modifier = Modifier.size(48.dp),
                    ) {
                        Icon(
                            imageVector = if (memo.isStarred) {
                                Icons.Rounded.Star
                            } else {
                                Icons.Rounded.StarBorder
                            },
                            contentDescription = if (memo.isStarred) {
                                "Remove ${memo.title} from starred"
                            } else {
                                "Star ${memo.title}"
                            },
                            tint = if (memo.isStarred) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                MaterialTheme.colorScheme.onSurfaceVariant
                            },
                        )
                    }
                    IconButton(
                        onClick = onLongPress,
                        modifier = Modifier.size(48.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Rounded.MoreVert,
                            contentDescription = "More options for ${memo.title}",
                        )
                    }
                }
            }

            Text(
                text = memo.generatedSummary
                    ?: memo.transcriptPreview
                    ?: memo.transcription.statusLabel,
                style = MaterialTheme.typography.bodyMedium,
                color = if (selected) {
                    selectedContentColor.copy(alpha = 0.82f)
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
                maxLines = if (memo.generatedSummary != null) 1 else 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth(),
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(
                    onClick = onTogglePlayback,
                    modifier = Modifier.size(36.dp),
                ) {
                    Icon(
                        imageVector = if (memo.isPlaying) {
                            Icons.Rounded.Pause
                        } else {
                            Icons.Rounded.PlayArrow
                        },
                        contentDescription = if (memo.isPlaying) {
                            "Pause ${memo.title}"
                        } else {
                            "Play ${memo.title}"
                        },
                        modifier = Modifier.size(22.dp),
                    )
                }
                VoiceMemoWaveform(
                    amplitudes = memo.waveform,
                    progress = memo.playbackFraction,
                    description = "Waveform for ${memo.title}",
                    activeColor = if (selected) {
                        selectedContentColor
                    } else {
                        MaterialTheme.colorScheme.primary
                    },
                    inactiveColor = if (selected) {
                        selectedContentColor.copy(alpha = 0.55f)
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.65f)
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(22.dp),
                )
                Text(
                    text = memo.durationLabel,
                    style = MaterialTheme.typography.labelLarge,
                    color = if (selected) {
                        selectedContentColor
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
            }
        }
    }
}

@Composable
private fun LibraryEmptyState(
    hasQuery: Boolean,
    inProject: Boolean = false,
    starred: Boolean = false,
    unassigned: Boolean = false,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 64.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Icon(
            imageVector = if (hasQuery) Icons.Rounded.Info else Icons.Rounded.Mic,
            contentDescription = null,
            modifier = Modifier.size(48.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Text(
            text = when {
                hasQuery -> "No matching recordings"
                inProject -> "No recordings in this project"
                starred -> "No starred recordings"
                unassigned -> "No unassigned recordings"
                else -> "No recordings yet"
            },
            style = MaterialTheme.typography.headlineSmall,
        )
        Text(
            text = if (hasQuery) {
                "Try part of a word, another spelling, or fewer keywords."
            } else if (inProject) {
                "Record here to add a new memo, or select recordings and move them."
            } else if (starred) {
                "Tap the star on a recording to keep it in this view."
            } else if (unassigned) {
                "Every recording is already assigned to a project."
            } else {
                "Tap Record to capture a new memo, or import saved audio from Settings."
            },
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun SelectionActionsSheet(
    selectedCount: Int,
    onDismiss: () -> Unit,
    onDelete: () -> Unit,
    onShare: () -> Unit,
    onSelectAll: () -> Unit,
    onMove: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
        ) {
            Text(
                text = "$selectedCount selected",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            )
            SelectionAction(
                label = "Move to project",
                icon = {
                    Icon(
                        imageVector = Icons.AutoMirrored.Rounded.DriveFileMove,
                        contentDescription = null,
                    )
                },
                onClick = onMove,
            )
            SelectionAction(
                label = "Delete",
                icon = {
                    Icon(
                        imageVector = Icons.Rounded.Delete,
                        contentDescription = null,
                    )
                },
                onClick = onDelete,
            )
            SelectionAction(
                label = "Share",
                icon = {
                    Icon(
                        imageVector = Icons.Rounded.Share,
                        contentDescription = null,
                    )
                },
                onClick = onShare,
            )
            SelectionAction(
                label = "Select all",
                icon = {
                    Icon(
                        imageVector = Icons.Rounded.SelectAll,
                        contentDescription = null,
                    )
                },
                onClick = onSelectAll,
            )
            Spacer(Modifier.height(8.dp))
            TextButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.End)
                    .padding(horizontal = 16.dp),
            ) {
                Text("Cancel")
            }
        }
    }
}

@Composable
private fun MoveToProjectSheet(
    projects: List<ProjectListItemUi>,
    onDismiss: () -> Unit,
    onSelect: (String?) -> Unit,
    onCreate: () -> Unit,
) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp),
        ) {
            Text(
                text = "Move to project",
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            )
            SelectionAction(
                label = "New project",
                icon = { Icon(Icons.Rounded.Add, contentDescription = null) },
                onClick = onCreate,
            )
            SelectionAction(
                label = "No project",
                icon = { Icon(Icons.Rounded.Close, contentDescription = null) },
                onClick = { onSelect(null) },
            )
            projects.forEach { project ->
                SelectionAction(
                    label = project.name,
                    icon = { Icon(Icons.Rounded.Folder, contentDescription = null) },
                    onClick = { onSelect(project.id) },
                )
            }
            Spacer(Modifier.height(8.dp))
            TextButton(
                onClick = onDismiss,
                modifier = Modifier
                    .align(Alignment.End)
                    .padding(horizontal = 16.dp),
            ) {
                Text("Cancel")
            }
        }
    }
}

@Composable
private fun CreateProjectDialog(
    onDismiss: () -> Unit,
    onCreate: (String) -> Unit,
) {
    var name by rememberSaveable { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = { Icon(Icons.Rounded.Folder, contentDescription = null) },
        title = { Text("New project") },
        text = {
            OutlinedTextField(
                value = name,
                onValueChange = { name = it.take(80) },
                label = { Text("Project name") },
                singleLine = true,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("project_name_field"),
            )
        },
        confirmButton = {
            TextButton(
                onClick = { onCreate(name.trim()) },
                enabled = name.isNotBlank(),
                modifier = Modifier.testTag("create_project_confirm"),
            ) {
                Text("Create")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    )
}

@Composable
private fun SelectionAction(
    label: String,
    icon: @Composable () -> Unit,
    onClick: () -> Unit,
) {
    ListItem(
        headlineContent = { Text(label) },
        leadingContent = icon,
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 64.dp)
            .clickable(onClick = onClick),
    )
}

private fun recordingSectionLabel(createdAtMillis: Long): String {
    if (createdAtMillis <= 0L) return "Recordings"
    val today = Calendar.getInstance()
    val recordingDay = Calendar.getInstance().apply {
        timeInMillis = createdAtMillis
    }
    if (today.isSameDay(recordingDay)) return "Today"
    today.add(Calendar.DAY_OF_YEAR, -1)
    if (today.isSameDay(recordingDay)) return "Yesterday"
    return SimpleDateFormat("MMMM d, yyyy", Locale.getDefault()).format(Date(createdAtMillis))
}

private fun Calendar.isSameDay(other: Calendar): Boolean =
    get(Calendar.ERA) == other.get(Calendar.ERA) &&
        get(Calendar.YEAR) == other.get(Calendar.YEAR) &&
        get(Calendar.DAY_OF_YEAR) == other.get(Calendar.DAY_OF_YEAR)

private val ProjectIcon.label: String
    get() = when (this) {
        ProjectIcon.WORK -> "Work"
        ProjectIcon.LIGHTBULB -> "Idea"
        ProjectIcon.PERSON -> "Personal"
        ProjectIcon.HOME -> "Home"
        ProjectIcon.SCHOOL -> "Learning"
        ProjectIcon.MUSIC -> "Music"
    }

private fun ProjectIcon.imageVector(): androidx.compose.ui.graphics.vector.ImageVector =
    when (this) {
        ProjectIcon.WORK -> Icons.Rounded.Work
        ProjectIcon.LIGHTBULB -> Icons.Rounded.Lightbulb
        ProjectIcon.PERSON -> Icons.Rounded.Person
        ProjectIcon.HOME -> Icons.Rounded.Home
        ProjectIcon.SCHOOL -> Icons.Rounded.School
        ProjectIcon.MUSIC -> Icons.Rounded.MusicNote
    }
