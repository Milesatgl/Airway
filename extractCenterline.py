from pathlib import Path
import vtk, logging, sys
from vtk import vtkPolyData, vtkDataArray, vtkPoints
# from vmtk import vtkvmtk
import vmtk.vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
import vmtk.vtkvmtkMiscPython as vtkvmtkMisc
import vmtk.vtkvmtkDifferentialGeometryPython as vtkvmtkDifferentialGeometry
from typing import Sequence

#region--------------------------logging------------------------------------
FMT = '%(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%m-%d %H:%M:%S'
class CustomLog:
    def __init__(
            self, logFilePath: str, loggerName: str, logLevel = 'info'
    ):
        self.logger = logging.getLogger(name=loggerName)
        if self.logger.handlers:
            self.logger.handlers.clear()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.logger.setLevel(logLevel.upper())
        self.logFilePath = logFilePath

        fh = self.getFileHandler(self.logFilePath)
        self.logger.addHandler(fh)
        ch = self.getConsoleHandler()
        self.logger.addHandler(ch)
    
    def getFileHandler(self, fileName):
        fileHandler = logging.FileHandler(fileName, encoding='utf-8', mode='a')
        fileHandler.setFormatter(self.formatter)
        return fileHandler
    
    def getConsoleHandler(self):
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(self.formatter)
        return consoleHandler

#endregion------------------------------------------------------------------

def read_stl_file(file_path) -> vtkPolyData | None:
    '''-> vtkPolyData'''
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def writePolyData(outPath: str, polyData: vtkPolyData):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outPath)
    writer.SetInputData(polyData)
    writer.SetDataModeToAscii()
    writer.Write()

# 定义一个类来处理点的交互选择
class PointPickerInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event) # type: ignore
        self.picker = vtk.vtkPointPicker()
        self.point = None

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())
        self.point = self.picker.GetPickPosition()
        LOGGER.info(f"Picked point: {self.point}")
        self.OnLeftButtonDown()
        return


# 定义函数来通过图形界面选择入口点
def select_entry_point_gui(surface):
    # 设置渲染器和渲染窗口
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    # 设置STL模型的映射
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    
    # 添加选择点的交互模式
    style = PointPickerInteractorStyle()
    style.SetDefaultRenderer(renderer)
    render_window_interactor.SetInteractorStyle(style)
    
    # 开始渲染并交互
    render_window.Render()
    print("Please click on the entry point on the model.")
    render_window_interactor.Start()
    
    return style.point


class ExtractCenterlineLogic:
    def __init__(self):
        # ScriptedLoadableModuleLogic.__init__(self)
        self.blankingArrayName = 'Blanking'
        self.radiusArrayName = 'Radius'  # maximum inscribed sphere radius
        self.groupIdsArrayName = 'GroupIds'
        self.centerlineIdsArrayName = 'CenterlineIds'
        self.tractIdsArrayName = 'TractIds'
        self.topologyArrayName = 'Topology'
        self.marksArrayName = 'Marks'
        self.lengthArrayName = 'Length'
        self.curvatureArrayName = 'Curvature'
        self.torsionArrayName = 'Torsion'
        self.tortuosityArrayName = 'Tortuosity'
        self.frenetTangentArrayName = 'FrenetTangent'
        self.frenetNormalArrayName = 'FrenetNormal'
        self.frenetBinormalArrayName = 'FrenetBinormal'
        self.TargetNumberOfPoints = 5000
        self.DecimationAggressiveness = 4.0
        self.PreprocessInputSurface = True
        self.SubdivideInputSurface = False
        self.CurveSamplingDistance = 1.0
    
    def preprocess(self, surfacePolyData: vtkPolyData, subdivide: bool) -> vtkPolyData:
        numberOfInputPoints = surfacePolyData.GetNumberOfPoints()
        if numberOfInputPoints == 0:
            raise ValueError('Input surface model is empty')
        #TODO: This depends on slicer
        # reductionFactor = (numberOfInputPoints - targetNumberOfPoints) / numberOfInputPoints
        # if reductionFactor > 0:

        surfaceTrianglelator = vtk.vtkTriangleFilter()
        surfaceTrianglelator.SetInputData(surfacePolyData)
        surfaceTrianglelator.PassLinesOff()
        surfaceTrianglelator.PassVertsOff()
        surfaceTrianglelator.Update()

        if subdivide:
            subdiv = vtk.vtkLinearSubdivisionFilter()
            subdiv.SetInputData(surfaceTrianglelator.GetOutput())
            subdiv.SetNumberOfSubdivisions(1)
            subdiv.Update()
            if subdiv.GetOutput().GetNumberOfPoints() == 0:
                LOGGER.warning(f"Mesh subdivision failed. Skip subdivision step.")
                subdivide = False
        normals = vtk.vtkPolyDataNormals()
        if subdivide:
            normals.SetInputData(subdiv.GetOutput())
        else:
            normals.SetInputData(surfaceTrianglelator.GetOutput())
        normals.SetAutoOrientNormals(1)
        normals.SetFlipNormals(0)
        normals.SetConsistency(1)
        normals.SplittingOff()
        normals.Update()
        return normals.GetOutput()
    
    def openSurfaceAtPoint(self, polyData: vtkPolyData, holePosition: Sequence[float], holePointIndex=None):
        '''
        Modifies the polyData by cutting a hole at the given position.
        '''
        if holePointIndex is None:
            pointLocator = vtk.vtkPointLocator()
            pointLocator.SetDataSet(polyData)
            pointLocator.BuildLocator()
            # find the closest point to the desired hole position
            holePointIndex = pointLocator.FindClosestPoint(holePosition)

        if holePointIndex < 0:
            # Calling GetPoint(-1) would crash the application
            raise ValueError("openSurfaceAtPoint failed: empty input polydata")

        # Tell the polydata to build 'upward' links from points to cells
        polyData.BuildLinks()
        # Mark cells as deleted
        cellIds = vtk.vtkIdList()
        polyData.GetPointCells(holePointIndex, cellIds)
        removeFirstCell = True
        if removeFirstCell:
            # remove first cell only (smaller hole)
            if cellIds.GetNumberOfIds() > 0:
                polyData.DeleteCell(cellIds.GetId(0))
                polyData.RemoveDeletedCells()
        else:
            # remove all cells
            for cellIdIndex in range(cellIds.GetNumberOfIds()):
                polyData.DeleteCell(cellIds.GetId(cellIdIndex))
            polyData.RemoveDeletedCells()


    def extractNetWork(self, surfacePolyData: vtkPolyData, startPosition: Sequence[float] | None, computeGeometry=False):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(surfacePolyData)
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(cleaner.GetOutputPort())
        triangleFilter.Update()
        simplifiedPolyData: vtkPolyData = triangleFilter.GetOutput()
        if startPosition is None:
            
            # If no endpoints are specific then use the closest point to a corner
            bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            simplifiedPolyData.GetBounds(bounds)
            startPosition = [bounds[0], bounds[2], bounds[4]]
            LOGGER.debug(f'Start position is none, calculated position is: {startPosition}')

        self.openSurfaceAtPoint(simplifiedPolyData, startPosition)

        # Extract network
        networkExtraction = vtkvmtkMisc.vtkvmtkPolyDataNetworkExtraction()
        networkExtraction.SetInputData(simplifiedPolyData)
        networkExtraction.SetAdvancementRatio(1.05)
        networkExtraction.SetRadiusArrayName(self.radiusArrayName)
        networkExtraction.SetTopologyArrayName(self.topologyArrayName)
        networkExtraction.SetMarksArrayName(self.marksArrayName)
        networkExtraction.Update()

        if computeGeometry:
            centerlineGeometry = vtkvmtkComputationalGeometry.vtkvmtkCenterlineGeometry()
            centerlineGeometry.SetInputData(networkExtraction.GetOutput())
            centerlineGeometry.SetLengthArrayName(self.lengthArrayName)
            centerlineGeometry.SetCurvatureArrayName(self.curvatureArrayName)
            centerlineGeometry.SetTorsionArrayName(self.torsionArrayName)
            centerlineGeometry.SetTortuosityArrayName(self.tortuosityArrayName)
            centerlineGeometry.SetFrenetTangentArrayName(self.frenetTangentArrayName)
            centerlineGeometry.SetFrenetNormalArrayName(self.frenetNormalArrayName)
            centerlineGeometry.SetFrenetBinormalArrayName(self.frenetBinormalArrayName)
            # centerlineGeometry.SetLineSmoothing(0)
            # centerlineGeometry.SetOutputSmoothedLines(0)
            # centerlineGeometry.SetNumberOfSmoothingIterations(100)
            # centerlineGeometry.SetSmoothingFactor(0.1)
            centerlineGeometry.Update()
            return centerlineGeometry.GetOutput()
        else:
            return networkExtraction.GetOutput()
        

    def getEndPoints(self, inputNetworkPolyData: vtkPolyData, startPointPosition) -> Sequence[tuple[float]]:
        '''
        Clips the surfacePolyData on the endpoints identified using the networkPolyData.
        If startPointPosition is specified then start point will be the closest point to that position.
        Returns list of endpoint positions. Largest radius point is be the first in the list.
        '''
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(inputNetworkPolyData)
        cleaner.Update()
        network: vtkPolyData = cleaner.GetOutput()
        network.BuildCells()
        network.BuildLinks(0)

        networkPoints: vtkPoints = network.GetPoints()
        radiusArray: vtkDataArray = network.GetPointData().GetArray(self.radiusArrayName)

        startPointId = -1
        maxRadius = 0
        minDistance2 = 0

        endpointIds = vtk.vtkIdList()
        for i in range(network.GetNumberOfCells()):
            numberOfCellPoints = network.GetCell(i).GetNumberOfPoints()
            if numberOfCellPoints < 2:
                continue

            for pointIndex in [0, numberOfCellPoints - 1]:
                pointId = network.GetCell(i).GetPointId(pointIndex)
                pointCells = vtk.vtkIdList()
                network.GetPointCells(pointId, pointCells)
                if pointCells.GetNumberOfIds() == 1:
                    endpointIds.InsertUniqueId(pointId)
                    if startPointPosition is not None:
                        # find start point based on position
                        position = networkPoints.GetPoint(pointId)
                        distance2 = vtk.vtkMath.Distance2BetweenPoints(position, startPointPosition)
                        if startPointId < 0 or distance2 < minDistance2:
                            minDistance2 = distance2
                            startPointId = pointId
                    else:
                        # find start point based on radius
                        radius = radiusArray.GetValue(pointId) # type: ignore
                        # radius = radiusArray.GetVariantValue(pointId)
                        # radius = radiusArray.GetTuple1(pointId)
                        if startPointId < 0 or radius > maxRadius:
                            maxRadius = radius
                            startPointId = pointId
        
        endpointPositions = []
        numberOfEndpointIds = endpointIds.GetNumberOfIds()
        if numberOfEndpointIds == 0:
            return endpointPositions
        # add the largest radius point first
        endpointPositions.append(networkPoints.GetPoint(startPointId))
        # add all the other points
        for pointIdIndex in range(numberOfEndpointIds):
            pointId = endpointIds.GetId(pointIdIndex)
            if pointId == startPointId:
                # already added
                continue
            endpointPositions.append(networkPoints.GetPoint(pointId))

        return endpointPositions

    # def onAutoDetectEndPoints(self, surfacePolyData: vtkPolyData):
    #     # TODO: Start position configuration
    #     networkPolyData = self.extractNetWork(surfacePolyData=surfacePolyData, startPosition=None)
    #     return self.getEndPoints(networkPolyData, None)

    def extractNonManifoldEdges(self, polyData, nonManifoldEdgesPolyData=None):
        '''
        Returns non-manifold edge center positions.
        nonManifoldEdgesPolyData: optional vtk.vtkPolyData() input, if specified then a polydata is returned that contains the edges
        '''
        
        neighborhoods = vtkvmtkDifferentialGeometry.vtkvmtkNeighborhoods()
        neighborhoods.SetNeighborhoodTypeToPolyDataManifoldNeighborhood()
        neighborhoods.SetDataSet(polyData)
        neighborhoods.Build()

        polyData.BuildCells()
        polyData.BuildLinks(0)

        edgeCenterPositions = []

        neighborCellIds = vtk.vtkIdList()
        nonManifoldEdgeLines = vtk.vtkCellArray()
        points = polyData.GetPoints()
        for i in range(neighborhoods.GetNumberOfNeighborhoods()):
            neighborhood = neighborhoods.GetNeighborhood(i)
            for j in range(neighborhood.GetNumberOfPoints()):
                neighborId = neighborhood.GetPointId(j)
                if i < neighborId:
                    neighborCellIds.Initialize()
                    polyData.GetCellEdgeNeighbors(-1, i, neighborId, neighborCellIds)
                    if neighborCellIds.GetNumberOfIds() > 2:
                        nonManifoldEdgeLines.InsertNextCell(2)
                        nonManifoldEdgeLines.InsertCellPoint(i)
                        nonManifoldEdgeLines.InsertCellPoint(neighborId)
                        p1 = points.GetPoint(i)
                        p2 = points.GetPoint(neighborId)
                        edgeCenterPositions.append([(p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0, (p1[2]+p2[2])/2.0])

        if nonManifoldEdgesPolyData:
            if not polyData.GetPoints():
                raise ValueError("Failed to get non-manifold edges (neighborhood filter output was empty)")
            pointsCopy = vtk.vtkPoints()
            pointsCopy.DeepCopy(polyData.GetPoints())
            nonManifoldEdgesPolyData.SetPoints(pointsCopy)
            nonManifoldEdgesPolyData.SetLines(nonManifoldEdgeLines)

        return edgeCenterPositions
    
    def extractCenterline(self, surfacePolyData, endPoints: Sequence[tuple[float]],curveSamplingDistance=1.0):
        """Compute centerline.
        This is more robust and accurate but takes longer than the network extraction.
        :param surfacePolyData:
        :param endPointsMarkupsNode:
        :return:
        """
        # Cap all the holes that are in the mesh that are not marked as endpoints
        # Maybe this is not needed.
        capDisplacement = 0.0
        surfaceCapper = vtkvmtkComputationalGeometry.vtkvmtkCapPolyData()
        surfaceCapper.SetInputData(surfacePolyData)
        surfaceCapper.SetDisplacement(capDisplacement)
        surfaceCapper.SetInPlaneDisplacement(capDisplacement)
        surfaceCapper.Update()

        tubePolyData = surfaceCapper.GetOutput()
        # pos = [0.0, 0.0, 0.0]
        # It seems that vtkvmtkComputationalGeometry does not need holes (unlike network extraction, which does need one hole)
        # # Punch holes at surface endpoints to have tubular structure
        # tubePolyData = surfaceCapper.GetOutput()
        # numberOfEndpoints = endPointsMarkupsNode.GetNumberOfControlPoints()
        # for pointIndex in range(numberOfEndpoints):
        #     endPointsMarkupsNode.GetNthControlPointPosition(pointIndex, pos)
        #     self.openSurfaceAtPoint(tubePolyData, pos)

        sourceIdList = vtk.vtkIdList()
        targetIdList = vtk.vtkIdList()

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(tubePolyData)
        pointLocator.BuildLocator()

        for controlPointIndex in range(len(endPoints)):
            ptPose = endPoints[controlPointIndex]
            pointId = pointLocator.FindClosestPoint(ptPose)
            if controlPointIndex == 0:
                sourceIdList.InsertNextId(pointId)
            else:
                targetIdList.InsertNextId(pointId)

        centerlineFilter = vtkvmtkComputationalGeometry.vtkvmtkPolyDataCenterlines()
        centerlineFilter.SetInputData(tubePolyData)
        centerlineFilter.SetSourceSeedIds(sourceIdList)
        centerlineFilter.SetTargetSeedIds(targetIdList)
        centerlineFilter.SetRadiusArrayName(self.radiusArrayName)
        centerlineFilter.SetCostFunction('1/R')  # this makes path search prefer go through points with large radius
        centerlineFilter.SetFlipNormals(False)
        centerlineFilter.SetAppendEndPointsToCenterlines(0)

        # Voronoi smoothing slightly improves connectivity
        # Unfortunately, Voronoi smoothing is broken if VMTK is used with VTK9, therefore
        # disable this feature for now (https://github.com/vmtk/SlicerExtension-VMTK/issues/34)
        enableVoronoiSmoothing = True
        centerlineFilter.SetSimplifyVoronoi(enableVoronoiSmoothing)

        centerlineFilter.SetCenterlineResampling(0)
        centerlineFilter.SetResamplingStepLength(curveSamplingDistance)
        centerlineFilter.Update()

        if not centerlineFilter.GetOutput():
            raise ValueError("Failed to compute centerline (no output was generated)")
        centerlinePolyData = vtk.vtkPolyData()
        centerlinePolyData.DeepCopy(centerlineFilter.GetOutput())

        if not centerlineFilter.GetVoronoiDiagram():
            raise ValueError("Failed to compute centerline (no Voronoi diagram was generated)")
        voronoiDiagramPolyData = vtk.vtkPolyData()
        voronoiDiagramPolyData.DeepCopy(centerlineFilter.GetVoronoiDiagram())

        LOGGER.debug("End of Centerline Computation.")
        return centerlinePolyData, voronoiDiagramPolyData


if __name__ == '__main__':
    # 读取STL文件
    from datetime import datetime
    start = datetime.now()
    vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)
    logDir = Path(__file__).parent.joinpath('logs')
    logDir.mkdir(exist_ok=True)
    LOGGER = CustomLog(
        logFilePath=str(logDir / f'{start.strftime("%Y%m%d%H%M%S")}.log'),
        loggerName='CenterlineExtraction',
        logLevel='debug'
    ).logger

    stl_path = Path(r'E:\conda_projects\skeletonAnalysis\data\airTree.stl')
    inputSurfacePolyData = read_stl_file(str(stl_path))

    if not inputSurfacePolyData or inputSurfacePolyData.GetNumberOfPoints() == 0:
        raise ValueError('Invalid input surface')
    centerLineLogic = ExtractCenterlineLogic()
    preprocessedPolyData = centerLineLogic.preprocess(inputSurfacePolyData, subdivide=False)

    nonManifoldEdgePositions = centerLineLogic.extractNonManifoldEdges(preprocessedPolyData)
    numberOfNonManifoldEdges = len(nonManifoldEdgePositions)
    if numberOfNonManifoldEdges > 0:
        LOGGER.warning(("Found {0} non-manifold edges.").format(numberOfNonManifoldEdges) + '\n' + " Centerline computation may fail. Try to increase target point count or reduce decimation aggressiveness")

    startPosition = select_entry_point_gui(preprocessedPolyData)
    LOGGER.debug(f'Choosed start position at: {startPosition}')
    networkPolyData = centerLineLogic.extractNetWork(preprocessedPolyData, startPosition)
    endPoints = centerLineLogic.getEndPoints(networkPolyData, startPosition)

    centerlinePolyData, voronoiDiagramPolyData = centerLineLogic.extractCenterline(preprocessedPolyData, endPoints)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    # 设置STL模型的映射
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(centerlinePolyData)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    render_window.Render()
    render_window_interactor.Start()

    outCenterLline = stl_path.parent / f"{stl_path.stem}_centerline.vtp"
    outNetwork = stl_path.parent / f"{stl_path.stem}_network.vtp"
    outVoro = stl_path.parent / f"{stl_path.stem}_voronoiDiagram.vtp"
    writePolyData(str(outCenterLline), centerlinePolyData)
    writePolyData(str(outNetwork), networkPolyData)
    writePolyData(str(outVoro), voronoiDiagramPolyData)

    LOGGER.debug(f'Finished in {datetime.now() - start}')

    